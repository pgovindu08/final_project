"""Benchmark harness for FoodRoute (TS).

Times any callable on growing input sizes with warm-up and repeated runs,
and writes median/min/max results to CSV. Swap in SN's real methods when ready.
"""
import csv
import statistics
import time
from typing import Callable, Iterable


def time_once(fn: Callable, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def benchmark(fn: Callable, args: tuple, repeats: int = 7, warmup: int = 1) -> dict:
    for _ in range(warmup):
        fn(*args)
    times = [time_once(fn, *args) for _ in range(repeats)]
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "repeats": repeats,
    }


def run_suite(methods: dict, sizes: Iterable[int], make_args: Callable, out_csv: str,
              repeats: int = 7) -> None:
    """methods: {name: callable}; make_args(size) -> tuple of args for the callables."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "size", "median_s", "min_s", "max_s", "repeats"])
        for size in sizes:
            args = make_args(size)
            for name, fn in methods.items():
                r = benchmark(fn, args, repeats=repeats)
                w.writerow([name, size, r["median_s"], r["min_s"], r["max_s"], r["repeats"]])
                print(f"{name:20s} size={size:<8d} median={r['median_s']:.6f}s")


def check_tract_ids(ids: Iterable) -> list:
    """Return tract IDs that are not valid 11-digit strings (catches lost leading zeros)."""
    return [i for i in ids if not (str(i).isdigit() and len(str(i)) == 11)]


if __name__ == "__main__":
    # Placeholder methods so the harness runs before SN's code exists.
    def stub_repeated(n):  # stands in for O(N * work)
        return sum(i * i for i in range(n * 50))

    def stub_multi(n):  # stands in for single-pass work
        return sum(i * i for i in range(n))

    run_suite({"repeated_stub": stub_repeated, "multi_stub": stub_multi},
              sizes=[100, 200, 400, 800],
              make_args=lambda n: (n,),
              out_csv="benchmark_results.csv")
    print("bad ids:", check_tract_ids(["47093000100", "4709300100", 47093000200]))
