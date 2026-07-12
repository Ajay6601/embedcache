// Package rediscache is a tiny, dependency-free Redis client that backs a
// shared cache tier so multiple embedcache instances share entries instead of
// each keeping its own cold cache. It speaks just enough RESP (GET, SET with
// expiry, PING, AUTH, SELECT) over plain TCP to be the L2 behind the in-memory
// cache — no external Redis library, keeping the core stdlib-only.
//
// Values are the exact cached bytes (the upstream's raw embedding JSON) plus a
// compact varint header for the attributed token count, so a shared hit is
// byte-identical to a local hit. Redis holds the TTL, so expiry is handled by
// Redis for the shared copy while the in-memory tier keeps its own bound.
package rediscache

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"net"
	"strconv"
	"sync"
	"time"

	"github.com/Ajay6601/embedcache/internal/cache"
)

// Client is a small pooled Redis client implementing cache.Shared.
type Client struct {
	addr     string
	password string
	db       int
	prefix   string
	ttl      time.Duration
	dialTO   time.Duration
	ioTO     time.Duration

	mu   sync.Mutex
	pool []*conn
	max  int
}

type conn struct {
	nc net.Conn
	br *bufio.Reader
}

// Options configures the shared Redis tier.
type Options struct {
	Addr     string        // host:port
	Password string        // "" for none
	DB       int           // logical db
	Prefix   string        // key namespace, e.g. "ec:"
	TTL      time.Duration // 0 = no expiry on shared entries
	PoolSize int           // max pooled connections (default 8)
}

// New connects-checks the server once and returns a ready client. A failed
// initial PING is returned as an error so serve can fail fast on a bad -shared-redis.
func New(o Options) (*Client, error) {
	if o.PoolSize <= 0 {
		o.PoolSize = 8
	}
	c := &Client{
		addr:     o.Addr,
		password: o.Password,
		db:       o.DB,
		prefix:   o.Prefix,
		ttl:      o.TTL,
		dialTO:   3 * time.Second,
		ioTO:     3 * time.Second,
		max:      o.PoolSize,
	}
	cn, err := c.dial()
	if err != nil {
		return nil, err
	}
	if err := c.ping(cn); err != nil {
		cn.nc.Close()
		return nil, err
	}
	c.put(cn)
	return c, nil
}

func (c *Client) dial() (*conn, error) {
	nc, err := net.DialTimeout("tcp", c.addr, c.dialTO)
	if err != nil {
		return nil, err
	}
	cn := &conn{nc: nc, br: bufio.NewReader(nc)}
	if c.password != "" {
		if err := c.do(cn, nil, "AUTH", c.password); err != nil {
			nc.Close()
			return nil, err
		}
	}
	if c.db != 0 {
		if err := c.do(cn, nil, "SELECT", strconv.Itoa(c.db)); err != nil {
			nc.Close()
			return nil, err
		}
	}
	return cn, nil
}

func (c *Client) get() (*conn, error) {
	c.mu.Lock()
	if n := len(c.pool); n > 0 {
		cn := c.pool[n-1]
		c.pool = c.pool[:n-1]
		c.mu.Unlock()
		return cn, nil
	}
	c.mu.Unlock()
	return c.dial()
}

func (c *Client) put(cn *conn) {
	c.mu.Lock()
	if len(c.pool) < c.max {
		c.pool = append(c.pool, cn)
		c.mu.Unlock()
		return
	}
	c.mu.Unlock()
	cn.nc.Close()
}

func (c *Client) ping(cn *conn) error { return c.do(cn, nil, "PING") }

// Get implements cache.Shared. A miss (key absent) returns ok=false with no error.
func (c *Client) Get(key string) (cache.Entry, bool) {
	cn, err := c.get()
	if err != nil {
		return cache.Entry{}, false
	}
	var val []byte
	if err := c.do(cn, &val, "GET", c.prefix+key); err != nil {
		cn.nc.Close() // drop a broken connection instead of pooling it
		return cache.Entry{}, false
	}
	c.put(cn)
	if val == nil {
		return cache.Entry{}, false
	}
	return decodeEntry(val)
}

// Set implements cache.Shared, write-through with the configured TTL.
func (c *Client) Set(key string, e cache.Entry) {
	cn, err := c.get()
	if err != nil {
		return
	}
	val := encodeEntry(e)
	var args []string
	if c.ttl > 0 {
		args = []string{"SET", c.prefix + key, string(val), "PX", strconv.FormatInt(c.ttl.Milliseconds(), 10)}
	} else {
		args = []string{"SET", c.prefix + key, string(val)}
	}
	if err := c.do(cn, nil, args...); err != nil {
		cn.nc.Close()
		return
	}
	c.put(cn)
}

// ---- compact entry encoding: varint(tokens) varint(expiresAt) rawbytes ----

func encodeEntry(e cache.Entry) []byte {
	buf := make([]byte, 0, len(e.Raw)+2*binary.MaxVarintLen64)
	var tmp [binary.MaxVarintLen64]byte
	n := binary.PutVarint(tmp[:], int64(e.Tokens))
	buf = append(buf, tmp[:n]...)
	n = binary.PutVarint(tmp[:], e.ExpiresAt)
	buf = append(buf, tmp[:n]...)
	return append(buf, e.Raw...)
}

func decodeEntry(b []byte) (cache.Entry, bool) {
	tokens, n := binary.Varint(b)
	if n <= 0 {
		return cache.Entry{}, false
	}
	b = b[n:]
	exp, m := binary.Varint(b)
	if m <= 0 {
		return cache.Entry{}, false
	}
	raw := append([]byte(nil), b[m:]...)
	return cache.Entry{Raw: raw, Tokens: int(tokens), ExpiresAt: exp}, true
}

// ---- minimal RESP ----

// do writes a command and parses the reply. If out is non-nil it receives a
// bulk-string body (nil for a Redis null reply); other reply types are accepted
// only as success markers (+OK, PONG) or surfaced as errors.
func (c *Client) do(cn *conn, out *[]byte, args ...string) error {
	cn.nc.SetDeadline(time.Now().Add(c.ioTO))
	if err := writeCommand(cn.nc, args); err != nil {
		return err
	}
	line, err := cn.br.ReadString('\n')
	if err != nil {
		return err
	}
	if len(line) < 3 {
		return fmt.Errorf("short reply %q", line)
	}
	body := line[1 : len(line)-2] // strip type byte and CRLF
	switch line[0] {
	case '+', ':': // simple string / integer
		return nil
	case '-': // error
		return fmt.Errorf("redis: %s", body)
	case '$': // bulk string
		n, err := strconv.Atoi(body)
		if err != nil {
			return fmt.Errorf("bad bulk length %q", body)
		}
		if n < 0 { // null
			if out != nil {
				*out = nil
			}
			return nil
		}
		buf := make([]byte, n+2) // include trailing CRLF
		if _, err := readFull(cn.br, buf); err != nil {
			return err
		}
		if out != nil {
			*out = buf[:n]
		}
		return nil
	default:
		return fmt.Errorf("unexpected reply type %q", string(line[0]))
	}
}

func writeCommand(w net.Conn, args []string) error {
	buf := make([]byte, 0, 32)
	buf = append(buf, '*')
	buf = strconv.AppendInt(buf, int64(len(args)), 10)
	buf = append(buf, '\r', '\n')
	for _, a := range args {
		buf = append(buf, '$')
		buf = strconv.AppendInt(buf, int64(len(a)), 10)
		buf = append(buf, '\r', '\n')
		buf = append(buf, a...)
		buf = append(buf, '\r', '\n')
	}
	_, err := w.Write(buf)
	return err
}

func readFull(br *bufio.Reader, buf []byte) (int, error) {
	total := 0
	for total < len(buf) {
		n, err := br.Read(buf[total:])
		total += n
		if err != nil {
			return total, err
		}
	}
	return total, nil
}
