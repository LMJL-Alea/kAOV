"""
Toy profiling script for a kAOV analysis.

For each input size, it:
  1. creates a synthetic dataset with kaov.testing.dummy_data,
  2. builds an AOV object with AOV.from_formula,
  3. runs AOV.test(),
and reports wall-clock time, peak traced Python memory and peak RSS.

Run with:  python profile_kaov.py
"""

import gc
import itertools
import resource
import time
import tracemalloc

import pandas as pd

from kaov import AOV
from kaov.testing import dummy_data

# Input sizes to profile: (n_row, n_col)
# n_row x n_col = [100, 500, 1000, 5000, 10000] x [10, 50, 100, 500, 1000]
SIZES = list(itertools.product(
    [100, 500, 1000], #, 5000, 10000],   # n_row
    [10, 50, 100, 500] #, 1000],        # n_col
))
N_FACTOR = 2
N_GROUP = [3, 4]
N_TRUNC = 20          # truncation level for test()
SEED = 42


def peak_rss_mb():
    """Peak resident set size of the current process, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux


def run_once(n_row, n_col, nystrom=False, n_landmarks=None):
    """Run one full kAOV analysis and return (elapsed_s, peak_traced_mb, peak_rss_mb)."""
    gc.collect()
    rss_before = peak_rss_mb()

    tracemalloc.start()
    t0 = time.perf_counter()

    # 1. dummy dataset
    data, input_formula, output_formula = dummy_data(
        n_row=n_row, n_col=n_col, n_factor=N_FACTOR, n_group=N_GROUP, seed=SEED
    )
    # 2. AOV object from formula
    kfit = AOV.from_formula(
        f"{output_formula} ~ {input_formula}",
        data=data, nystrom=nystrom, n_landmarks=n_landmarks,
        random_gen=SEED, verbose=0,
    )
    # 3. hypothesis test
    res = kfit.test(n_trunc=N_TRUNC, verbose=0)

    elapsed = time.perf_counter() - t0
    _, peak_traced = tracemalloc.get_traced_memory()   # (current, peak) in bytes
    tracemalloc.stop()

    peak_traced_mb = peak_traced / 1e6
    peak_rss = peak_rss_mb()
    return elapsed, peak_traced_mb, peak_rss, res


def main():
    print(f"{'n_row':>6} | {'n_col':>5} | {'nystrom':>7} | "
          f"{'time (s)':>9} | {'traced peak (MB)':>17} | {'peak RSS (MB)':>14}")
    print("-" * 75)
    for n_row, n_col in SIZES:
        elapsed, traced_mb, rss_mb, _ = run_once(n_row, n_col, nystrom=False)
        print(f"{n_row:>6} | {n_col:>5} | {'-':>7} | "
              f"{elapsed:>9.2f} | {traced_mb:>17.1f} | {rss_mb:>14.1f}")

    # optional: same sizes with the Nystrom approximation
    print("\nWith Nystrom approximation (n_landmarks = n_row // 5):")
    print(f"{'n_row':>6} | {'n_col':>5} | {'nystrom':>7} | "
          f"{'time (s)':>9} | {'traced peak (MB)':>17} | {'peak RSS (MB)':>14}")
    print("-" * 75)
    for n_row, n_col in SIZES:
        elapsed, traced_mb, rss_mb, _ = run_once(
            n_row, n_col, nystrom=True, n_landmarks=max(1, n_row // 5)
        )
        print(f"{n_row:>6} | {n_col:>5} | {'yes':>7} | "
              f"{elapsed:>9.2f} | {traced_mb:>17.1f} | {rss_mb:>14.1f}")


if __name__ == "__main__":
    main()

    # # Note: using cProfile
    # import cProfile, pstats
    # cProfile.run("run_once(1000, 100)", "prof.out")
    # pstats.Stats("prof.out").sort_stats("cumulative").print_stats(20)
