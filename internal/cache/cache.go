// Package cache is a sharded in-memory LRU for embedding entries with
// optional gob snapshot persistence.
package cache

import (
	"container/list"
	"encoding/gob"
	"os"
	"sync"
	"time"
)

// Entry is one cached embedding: the raw JSON value of the "embedding" field
// exactly as the upstream returned it, plus the token count attributed to
// computing it (used for savings accounting). ExpiresAt is unix nanoseconds;
// zero means the entry never expires.
type Entry struct {
	Raw       []byte
	Tokens    int
	ExpiresAt int64
}

const shardCount = 16

// Shared is an optional second-level cache (e.g. Redis) that multiple
// embedcache instances point at, so a vector one instance computed is reused by
// the whole fleet instead of every replica starting cold. Implementations must
// be safe for concurrent use; a miss returns ok=false.
type Shared interface {
	Get(key string) (Entry, bool)
	Set(key string, e Entry)
}

type Cache struct {
	shards             [shardCount]*shard
	maxEntriesPerShard int
	maxBytesPerShard   int64
	shared             Shared
}

type shard struct {
	mu    sync.Mutex
	ll    *list.List
	items map[string]*list.Element
	bytes int64
}

type lruItem struct {
	key   string
	entry Entry
}

// New creates a cache bounded by total entry count and total payload bytes.
// Zero means unbounded for that dimension.
func New(maxEntries int, maxBytes int64) *Cache {
	c := &Cache{
		maxEntriesPerShard: maxEntries / shardCount,
		maxBytesPerShard:   maxBytes / shardCount,
	}
	if maxEntries > 0 && c.maxEntriesPerShard == 0 {
		c.maxEntriesPerShard = 1
	}
	if maxBytes > 0 && c.maxBytesPerShard == 0 {
		c.maxBytesPerShard = 1
	}
	for i := range c.shards {
		c.shards[i] = &shard{ll: list.New(), items: map[string]*list.Element{}}
	}
	return c
}

// SetShared attaches a second-level shared cache (read-through on a local miss,
// write-through on Set). Pass nil to run purely in-memory.
func (c *Cache) SetShared(s Shared) { c.shared = s }

func (c *Cache) shardFor(key string) *shard {
	if len(key) == 0 {
		return c.shards[0]
	}
	return c.shards[int(key[0]^key[len(key)-1])%shardCount]
}

func (c *Cache) Get(key string) (Entry, bool) {
	s := c.shardFor(key)
	s.mu.Lock()
	el, ok := s.items[key]
	if ok {
		item := el.Value.(*lruItem)
		if item.entry.ExpiresAt > 0 && time.Now().UnixNano() > item.entry.ExpiresAt {
			s.ll.Remove(el)
			delete(s.items, key)
			s.bytes -= int64(len(item.entry.Raw))
			ok = false
		} else {
			s.ll.MoveToFront(el)
			e := item.entry
			s.mu.Unlock()
			return e, true
		}
	}
	s.mu.Unlock()

	// local miss: consult the shared tier and, on a hit, warm the local cache so
	// the next lookup is served in-process.
	if c.shared != nil {
		if e, hit := c.shared.Get(key); hit {
			c.setLocal(key, e)
			return e, true
		}
	}
	return Entry{}, false
}

// Set stores an entry locally and, if a shared tier is attached, write-through
// to it. The shared write is asynchronous so it never adds latency to the
// response path (the local copy is what this request serves).
func (c *Cache) Set(key string, e Entry) {
	c.setLocal(key, e)
	if c.shared != nil {
		go c.shared.Set(key, e)
	}
}

func (c *Cache) setLocal(key string, e Entry) {
	s := c.shardFor(key)
	s.mu.Lock()
	defer s.mu.Unlock()
	if el, ok := s.items[key]; ok {
		old := el.Value.(*lruItem)
		s.bytes += int64(len(e.Raw)) - int64(len(old.entry.Raw))
		old.entry = e
		s.ll.MoveToFront(el)
	} else {
		el := s.ll.PushFront(&lruItem{key: key, entry: e})
		s.items[key] = el
		s.bytes += int64(len(e.Raw))
	}
	for (c.maxEntriesPerShard > 0 && s.ll.Len() > c.maxEntriesPerShard) ||
		(c.maxBytesPerShard > 0 && s.bytes > c.maxBytesPerShard && s.ll.Len() > 1) {
		back := s.ll.Back()
		if back == nil {
			break
		}
		item := back.Value.(*lruItem)
		s.ll.Remove(back)
		delete(s.items, item.key)
		s.bytes -= int64(len(item.entry.Raw))
	}
}

// Flush drops every entry and returns how many were dropped. Used when the
// upstream model changes in place (same name, new weights) and cached vectors
// are no longer valid.
func (c *Cache) Flush() int {
	n := 0
	for _, s := range c.shards {
		s.mu.Lock()
		n += s.ll.Len()
		s.ll.Init()
		s.items = map[string]*list.Element{}
		s.bytes = 0
		s.mu.Unlock()
	}
	return n
}

func (c *Cache) Len() int {
	n := 0
	for _, s := range c.shards {
		s.mu.Lock()
		n += s.ll.Len()
		s.mu.Unlock()
	}
	return n
}

func (c *Cache) Bytes() int64 {
	var n int64
	for _, s := range c.shards {
		s.mu.Lock()
		n += s.bytes
		s.mu.Unlock()
	}
	return n
}

// Snapshot writes all entries to path atomically (write temp, rename).
func (c *Cache) Snapshot(path string) error {
	all := map[string]Entry{}
	for _, s := range c.shards {
		s.mu.Lock()
		for k, el := range s.items {
			all[k] = el.Value.(*lruItem).entry
		}
		s.mu.Unlock()
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(f).Encode(all); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	os.Remove(path) // Windows rename does not overwrite
	return os.Rename(tmp, path)
}

// Load restores entries from a snapshot file. A missing file is not an error.
func (c *Cache) Load(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	defer f.Close()
	var all map[string]Entry
	if err := gob.NewDecoder(f).Decode(&all); err != nil {
		return 0, err
	}
	for k, e := range all {
		c.Set(k, e)
	}
	return len(all), nil
}
