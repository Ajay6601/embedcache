# embedcache shared cache: cross-instance reuse (real Redis)

Generated 2026-07-12 17:49 EDT. Two separate embedcache processes, each with its own in-memory
cache, both pointed at one Redis (`127.0.0.1:6379`), in front of one backend that counts its
own calls. This is what keeps the savings from eroding across a multi-replica fleet.

- **PASS** — instance A computes a new input (miss): status=miss, backend calls so far=1
- **PASS** — instance B reuses instance A's vector from the shared tier (hit, no new backend call): status=hit, backend calls total=1 (still 1 = B did not recompute)
- **PASS** — the shared vector is byte-identical across instances: A embedding 2846 bytes, B embedding 2846 bytes, equal=true
- **PASS** — a genuinely new input still reaches the backend: status=miss, backend calls total=2

Net: across two independent instances, the backend computed each distinct input **once**.
A second replica does not restart cold; it inherits the whole fleet's cache through Redis.
