package budget

import (
	"testing"
	"time"
)

func TestDefaultLimitApplies(t *testing.T) {
	e := New(100, time.Hour)
	if ok, _ := e.Allow("Bearer sk-a"); !ok {
		t.Fatal("fresh key must be allowed")
	}
	e.Record("Bearer sk-a", 99)
	if ok, _ := e.Allow("Bearer sk-a"); !ok {
		t.Fatal("under budget must be allowed")
	}
	e.Record("Bearer sk-a", 1) // exactly at limit
	ok, retry := e.Allow("Bearer sk-a")
	if ok {
		t.Fatal("at limit must be rejected")
	}
	if retry <= 0 || retry > time.Hour {
		t.Fatalf("retryAfter out of range: %v", retry)
	}
}

func TestKeysAreIsolated(t *testing.T) {
	e := New(50, time.Hour)
	e.Record("Bearer sk-a", 50)
	if ok, _ := e.Allow("Bearer sk-a"); ok {
		t.Fatal("sk-a exhausted")
	}
	if ok, _ := e.Allow("Bearer sk-b"); !ok {
		t.Fatal("sk-b must be unaffected by sk-a's spend")
	}
}

func TestPerKeyOverrides(t *testing.T) {
	e := New(100, time.Hour)
	e.SetLimit("sk-small", 10)
	e.SetLimit("sk-vip", 0) // explicitly unlimited

	e.Record("Bearer sk-small", 10)
	if ok, _ := e.Allow("Bearer sk-small"); ok {
		t.Fatal("override limit must win over default")
	}
	e.Record("Bearer sk-vip", 1_000_000)
	if ok, _ := e.Allow("Bearer sk-vip"); !ok {
		t.Fatal("zero override means unlimited")
	}
}

func TestBearerPrefixAndBareKeyShareABucket(t *testing.T) {
	e := New(100, time.Hour)
	e.Record("Bearer sk-a", 60)
	e.Record("sk-a", 60) // azure-style bare credential, same key
	if ok, _ := e.Allow("Bearer sk-a"); ok {
		t.Fatal("Bearer-prefixed and bare forms of one key must share spend")
	}
}

func TestWindowResets(t *testing.T) {
	e := New(10, 40*time.Millisecond)
	e.Record("Bearer sk-a", 10)
	if ok, _ := e.Allow("Bearer sk-a"); ok {
		t.Fatal("exhausted")
	}
	time.Sleep(50 * time.Millisecond)
	if ok, _ := e.Allow("Bearer sk-a"); !ok {
		t.Fatal("window elapsed: budget must reset")
	}
	rem, limited := e.Remaining("Bearer sk-a")
	if !limited || rem != 10 {
		t.Fatalf("after reset remaining=%d limited=%v", rem, limited)
	}
}

func TestUnlimitedDefault(t *testing.T) {
	e := New(0, time.Hour)
	e.Record("Bearer sk-a", 1_000_000)
	if ok, _ := e.Allow("Bearer sk-a"); !ok {
		t.Fatal("no default limit means unlimited unless a key has an override")
	}
	if _, limited := e.Remaining("Bearer sk-a"); limited {
		t.Fatal("unlimited key must not report a remaining value")
	}
	e.SetLimit("sk-capped", 5)
	e.Record("Bearer sk-capped", 5)
	if ok, _ := e.Allow("Bearer sk-capped"); ok {
		t.Fatal("keyed limit must still enforce under an unlimited default")
	}
}

func TestNilEnforcerIsOff(t *testing.T) {
	var e *Enforcer
	if ok, _ := e.Allow("x"); !ok {
		t.Fatal("nil enforcer allows everything")
	}
	e.Record("x", 10) // must not panic
	if _, limited := e.Remaining("x"); limited {
		t.Fatal("nil enforcer has no limits")
	}
	if e.Report() != nil {
		t.Fatal("nil enforcer reports nothing")
	}
}

func TestReport(t *testing.T) {
	e := New(100, time.Hour)
	e.SetLimit("sk-a", 20)
	e.Record("Bearer sk-a", 5)
	e.Record("Bearer sk-b", 7) // default-limited key with spend
	rep := e.Report()
	if len(rep) != 2 {
		t.Fatalf("want 2 entries, got %d: %v", len(rep), rep)
	}
	for _, st := range rep {
		switch st.Limit {
		case 20:
			if st.Spent != 5 || st.Remaining != 15 {
				t.Fatalf("keyed entry wrong: %+v", st)
			}
		case 100:
			if st.Spent != 7 || st.Remaining != 93 {
				t.Fatalf("default entry wrong: %+v", st)
			}
		default:
			t.Fatalf("unexpected limit %d", st.Limit)
		}
		if st.ResetsInSecond <= 0 {
			t.Fatalf("resets_in_seconds must be positive: %+v", st)
		}
	}
}
