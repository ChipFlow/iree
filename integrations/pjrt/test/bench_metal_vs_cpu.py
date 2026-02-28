# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Metal vs CPU performance benchmark.

Runs JAX operations on both CPU and IREE Metal backends, reports timings,
speedups, and a summary table.

Usage:
    uv run python bench_metal_vs_cpu.py
    uv run python bench_metal_vs_cpu.py --num-iterations 5 --sizes small
    uv run python bench_metal_vs_cpu.py --output json > results.json
    uv run python bench_metal_vs_cpu.py --skip-sparse

Can also be run via pytest:
    uv run pytest bench_metal_vs_cpu.py -v
"""

import argparse
import json
import logging
import platform
import statistics
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------

SIZE_PRESETS = {
    "small": {
        "matmul": [128, 512],
        "elementwise": [1_000, 100_000],
        "reduction": [512],
        "batched_matmul": [128],
        "scan": [100, 500],
        "sparse": [20, 50],
    },
    "medium": {
        "matmul": [128, 512, 1024, 2048],
        "elementwise": [1_000, 100_000, 1_000_000],
        "reduction": [512, 2048],
        "batched_matmul": [128, 256],
        "scan": [100, 500, 1_000, 2_000],
        "sparse": [20, 50, 100],
    },
    "large": {
        "matmul": [128, 512, 1024, 2048, 4096],
        "elementwise": [1_000, 100_000, 1_000_000, 10_000_000],
        "reduction": [512, 2048, 4096],
        "batched_matmul": [128, 256],
        # Note: scan >2048 fails on Metal due to HAL resource allocation
        # limits. The FuseLoopIterationExecution pass cascades fusion up to
        # 2048 dispatches per execute; beyond that the loop is unfused and
        # Metal's per-iteration resource overhead causes OUT_OF_RANGE errors.
        "scan": [100, 500, 1_000, 2_000],
        "sparse": [20, 50, 100, 150],
    },
}


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

_BENCHMARKS: list[dict] = []


def benchmark(name: str):
    """Decorator to register a benchmark function.

    The decorated function receives (backend, sizes_dict, num_iterations)
    and must return a list of result dicts:
        [{"operation": str, "size": str, "times": [float, ...]}]
    Times are in seconds.
    """
    def decorator(fn):
        _BENCHMARKS.append({"name": name, "fn": fn})
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_fn(fn, num_iterations: int) -> list[float]:
    """Time *fn* over *num_iterations*, returning list of wall-clock seconds.

    Calls fn() once as warmup (result discarded from timings).
    Each call is synchronised via block_until_ready() if the result supports it.
    """
    # warmup / JIT compile
    result = fn()
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()

    times: list[float] = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        result = fn()
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------------------
# JAX benchmarks
# ---------------------------------------------------------------------------

@benchmark("matmul")
def bench_matmul(backend, sizes, num_iterations):
    import jax
    import jax.numpy as jnp

    results = []
    for n in sizes["matmul"]:
        with jax.default_device(jax.devices(backend)[0]):
            a = jnp.ones((n, n), dtype=jnp.float32)
            b = jnp.ones((n, n), dtype=jnp.float32)

            @jax.jit
            def matmul(x, y):
                return jnp.dot(x, y)

            times = time_fn(lambda: matmul(a, b), num_iterations)
        results.append({
            "operation": "matmul",
            "size": f"{n}x{n}",
            "times": times,
        })
    return results


@benchmark("elementwise")
def bench_elementwise(backend, sizes, num_iterations):
    import jax
    import jax.numpy as jnp

    results = []
    for n in sizes["elementwise"]:
        with jax.default_device(jax.devices(backend)[0]):
            x = jnp.linspace(0.0, 1.0, n, dtype=jnp.float32)

            @jax.jit
            def elemwise(v):
                return jnp.sin(v) * jnp.exp(-v)

            times = time_fn(lambda: elemwise(x), num_iterations)
        label = _human_count(n)
        results.append({
            "operation": "elementwise",
            "size": label,
            "times": times,
        })
    return results


@benchmark("reduction")
def bench_reduction(backend, sizes, num_iterations):
    import jax
    import jax.numpy as jnp

    results = []
    for n in sizes["reduction"]:
        with jax.default_device(jax.devices(backend)[0]):
            a = jnp.ones((n, n), dtype=jnp.float32)

            @jax.jit
            def reduce_sum(x):
                return jnp.sum(x)

            times = time_fn(lambda: reduce_sum(a), num_iterations)
        results.append({
            "operation": "reduction",
            "size": f"{n}x{n}",
            "times": times,
        })
    return results


@benchmark("batched_matmul")
def bench_batched_matmul(backend, sizes, num_iterations):
    import jax
    import jax.numpy as jnp

    results = []
    batch = 64
    for n in sizes["batched_matmul"]:
        with jax.default_device(jax.devices(backend)[0]):
            a = jnp.ones((batch, n, n), dtype=jnp.float32)
            b = jnp.ones((batch, n, n), dtype=jnp.float32)

            @jax.jit
            def batched_dot(x, y):
                return jax.vmap(jnp.dot)(x, y)

            times = time_fn(lambda: batched_dot(a, b), num_iterations)
        results.append({
            "operation": "batched_matmul",
            "size": f"64x{n}x{n}",
            "times": times,
        })
    return results


@benchmark("scan")
def bench_scan(backend, sizes, num_iterations):
    import jax
    import jax.numpy as jnp
    from jax import lax

    results = []
    for n in sizes["scan"]:
        with jax.default_device(jax.devices(backend)[0]):
            xs = jnp.ones(n, dtype=jnp.float32)

            @jax.jit
            def scan_sum(xs):
                def body(carry, x):
                    return carry + x, carry + x
                return lax.scan(body, jnp.float32(0.0), xs)

            try:
                times = time_fn(lambda: scan_sum(xs), num_iterations)
                results.append({
                    "operation": "scan",
                    "size": _human_count(n),
                    "times": times,
                })
            except Exception as e:
                logger.warning("scan benchmark skipped for n=%d: %s", n, e)
                results.append({
                    "operation": "scan",
                    "size": _human_count(n),
                    "times": None,
                    "error": str(e),
                })
    return results


# ---------------------------------------------------------------------------
# Sparse solver benchmark
# ---------------------------------------------------------------------------

def _sparse_available() -> bool:
    """Check whether jax.experimental.sparse.linalg.spsolve is importable."""
    try:
        from jax.experimental.sparse.linalg import spsolve  # noqa: F401
        return True
    except (ImportError, AttributeError):
        return False


def _create_poisson_2d(n: int):
    """Create 2D Poisson matrix (5-point stencil) of size n^2 x n^2."""
    import scipy.sparse as sp

    main_diag = -2 * np.ones(n)
    off_diag = np.ones(n - 1)
    T = sp.diags(
        [off_diag, main_diag, off_diag], [-1, 0, 1],
        shape=(n, n), format="csr",
    )
    I = sp.eye(n, format="csr")
    A = sp.kron(I, T) + sp.kron(T, I)
    return (-A).tocsr()


@benchmark("sparse_solve")
def bench_sparse_solve(backend, sizes, num_iterations):
    import jax
    import jax.numpy as jnp
    from jax.experimental.sparse.linalg import spsolve

    results = []
    for grid in sizes["sparse"]:
        N = grid * grid
        A = _create_poisson_2d(grid)

        x_true = np.ones(N, dtype=np.float32)
        b = (A @ x_true).astype(np.float32)

        with jax.default_device(jax.devices(backend)[0]):
            data_jax = jnp.array(A.data.astype(np.float32))
            indices_jax = jnp.array(A.indices.astype(np.int32))
            indptr_jax = jnp.array(A.indptr.astype(np.int32))
            b_jax = jnp.array(b)

            @jax.jit
            def solve(data, indices, indptr, rhs):
                return spsolve(data, indices, indptr, rhs)

            try:
                times = time_fn(
                    lambda: solve(data_jax, indices_jax, indptr_jax, b_jax),
                    num_iterations,
                )
                results.append({
                    "operation": "sparse_solve",
                    "size": f"{grid}x{grid}",
                    "times": times,
                })
            except Exception as e:
                logger.warning(
                    "sparse_solve benchmark failed for grid=%d: %s", grid, e,
                )
                results.append({
                    "operation": "sparse_solve",
                    "size": f"{grid}x{grid}",
                    "times": None,
                    "error": str(e),
                })
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def _median_ms(times: list[float]) -> float:
    return statistics.median(times) * 1000


def _min_ms(times: list[float]) -> float:
    return min(times) * 1000


def _max_ms(times: list[float]) -> float:
    return max(times) * 1000


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmarks(
    num_iterations: int,
    sizes: dict,
    skip_sparse: bool,
    backends: tuple[str, str] = ("cpu", "iree_metal"),
) -> list[dict]:
    """Run all registered benchmarks on both backends.

    Returns a list of result dicts with keys:
        operation, size, cpu_median_ms, metal_median_ms, speedup,
        cpu_min_ms, cpu_max_ms, metal_min_ms, metal_max_ms
    """
    import jax

    cpu_backend, metal_backend = backends

    # Verify backends are available
    try:
        jax.devices(cpu_backend)
    except RuntimeError:
        logger.error("CPU backend %r not available", cpu_backend)
        sys.exit(1)
    try:
        jax.devices(metal_backend)
    except RuntimeError:
        logger.error("Metal backend %r not available", metal_backend)
        sys.exit(1)

    combined: list[dict] = []

    for entry in _BENCHMARKS:
        name = entry["name"]
        fn = entry["fn"]

        if skip_sparse and name == "sparse_solve":
            logger.info("Skipping sparse_solve benchmarks (--skip-sparse)")
            continue

        if name == "sparse_solve" and not _sparse_available():
            logger.info("Skipping sparse_solve (spsolve not available)")
            continue

        logger.info("Running benchmark: %s", name)

        cpu_results = fn(cpu_backend, sizes, num_iterations)
        metal_results = fn(metal_backend, sizes, num_iterations)

        assert len(cpu_results) == len(metal_results), (
            f"Mismatch in result count for {name}"
        )

        for cpu_r, metal_r in zip(cpu_results, metal_results):
            assert cpu_r["operation"] == metal_r["operation"]
            assert cpu_r["size"] == metal_r["size"]

            row: dict = {
                "operation": cpu_r["operation"],
                "size": cpu_r["size"],
            }

            cpu_times = cpu_r.get("times")
            metal_times = metal_r.get("times")

            if cpu_times is None:
                row.update(cpu_median_ms=None, cpu_min_ms=None, cpu_max_ms=None)
                row["cpu_error"] = cpu_r.get("error", "unknown")
            else:
                row.update(
                    cpu_median_ms=_median_ms(cpu_times),
                    cpu_min_ms=_min_ms(cpu_times),
                    cpu_max_ms=_max_ms(cpu_times),
                )

            if metal_times is None:
                row.update(metal_median_ms=None, metal_min_ms=None, metal_max_ms=None)
                row["metal_error"] = metal_r.get("error", "unknown")
            else:
                row.update(
                    metal_median_ms=_median_ms(metal_times),
                    metal_min_ms=_min_ms(metal_times),
                    metal_max_ms=_max_ms(metal_times),
                )

            if cpu_times and metal_times:
                cpu_med = _median_ms(cpu_times)
                metal_med = _median_ms(metal_times)
                row["speedup"] = cpu_med / metal_med if metal_med > 0 else float("inf")
            else:
                row["speedup"] = None

            combined.append(row)

    return combined


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    """Print a human-readable results table."""
    print("\n=== Metal vs CPU Benchmark ===\n")

    header = (
        f"{'Operation':<25} {'Size':<12} {'CPU (ms)':>10} "
        f"{'Metal (ms)':>12} {'Speedup':>9}"
    )
    print(header)
    print("-" * len(header))

    metal_faster = 0
    total_comparable = 0
    max_speedup = 0.0
    max_speedup_label = ""

    for r in results:
        op = r["operation"]
        size = r["size"]
        cpu_ms = r.get("cpu_median_ms")
        metal_ms = r.get("metal_median_ms")
        speedup = r.get("speedup")

        cpu_str = f"{cpu_ms:10.2f}" if cpu_ms is not None else "     error"
        metal_str = f"{metal_ms:12.2f}" if metal_ms is not None else "       error"

        if speedup is not None:
            speedup_str = f"{speedup:8.1f}x"
            total_comparable += 1
            if speedup > 1.0:
                metal_faster += 1
            if speedup > max_speedup:
                max_speedup = speedup
                max_speedup_label = f"{op} {size}"
        else:
            speedup_str = "      n/a"

        print(f"{op:<25} {size:<12} {cpu_str} {metal_str} {speedup_str}")

    print()
    if total_comparable > 0:
        print(
            f"Summary: Metal faster in {metal_faster}/{total_comparable} benchmarks",
            end="",
        )
        if max_speedup > 1.0:
            print(f", max speedup {max_speedup:.1f}x ({max_speedup_label})")
        else:
            print()
    print()


def output_json(results: list[dict], num_iterations: int, sizes_name: str) -> None:
    """Print machine-readable JSON output."""
    payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "num_iterations": num_iterations,
        "sizes_preset": sizes_name,
        "results": results,
    }
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

def test_benchmark_runs():
    """Pytest-compatible entry: run a small benchmark and verify it completes."""
    import jax  # noqa: F401 — ensure JAX is importable

    sizes = SIZE_PRESETS["small"]
    results = run_benchmarks(
        num_iterations=2,
        sizes=sizes,
        skip_sparse=not _sparse_available(),
    )
    assert len(results) > 0, "No benchmark results produced"

    for r in results:
        assert "operation" in r
        assert "size" in r
        # At least one backend should have produced times
        assert (
            r.get("cpu_median_ms") is not None
            or r.get("metal_median_ms") is not None
            or r.get("cpu_error") is not None
            or r.get("metal_error") is not None
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Metal vs CPU performance benchmark",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=10,
        help="Number of timed iterations per benchmark (default: 10)",
    )
    parser.add_argument(
        "--sizes", choices=SIZE_PRESETS.keys(), default="medium",
        help="Size preset (default: medium)",
    )
    parser.add_argument(
        "--output", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--skip-sparse", action="store_true",
        help="Skip sparse solver benchmarks",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if platform.system() != "Darwin":
        logger.error("Metal benchmarks require macOS (got %s)", platform.system())
        sys.exit(1)

    # Configure JAX to have both backends available
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu,iree_metal")

    import jax  # noqa: E402 — import after env setup
    logger.info("JAX version: %s", jax.__version__)
    logger.info("Devices: %s", jax.devices())

    sizes = SIZE_PRESETS[args.sizes]

    results = run_benchmarks(
        num_iterations=args.num_iterations,
        sizes=sizes,
        skip_sparse=args.skip_sparse,
    )

    if args.output == "json":
        output_json(results, args.num_iterations, args.sizes)
    else:
        print_table(results)


if __name__ == "__main__":
    main()
