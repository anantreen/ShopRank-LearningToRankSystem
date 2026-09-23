#!/usr/bin/env python3
"""Send sequential requests and print p50/p95/p99 latency."""

import statistics
import sys
import time

try:
    import httpx
except ImportError:
    sys.exit("Install API extras first: pip install -e '.[api]'")


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]


if __name__ == "__main__":
    latencies = []
    for _ in range(100):
        started = time.perf_counter()
        response = httpx.post("http://127.0.0.1:8000/search", json={"query": "wireless gaming mouse"})
        response.raise_for_status()
        latencies.append((time.perf_counter() - started) * 1_000)
    print(f"mean={statistics.mean(latencies):.2f}ms p50={percentile(latencies,.50):.2f}ms "
          f"p95={percentile(latencies,.95):.2f}ms p99={percentile(latencies,.99):.2f}ms")

