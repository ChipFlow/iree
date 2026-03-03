"""Test VAJAX ring oscillator benchmark on Metal via IREE PJRT.

Requires: vajax optional dependency (uv sync --extra vajax)
Run: JAX_PLATFORMS=cpu,iree_metal uv run python -m pytest test_vajax_ring.py -v

Note: cpu platform is needed because vajax uses jax.devices("cpu") internally
for early collapse decisions during parsing.
"""

import platform
import sys

import pytest

# Skip on non-macOS
if platform.system() != "Darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)

try:
    from vajax.analysis import CircuitEngine
    from vajax.benchmarks.registry import get_benchmark
except ImportError:
    pytest.skip(
        "vajax not installed (run: uv sync --extra vajax)", allow_module_level=True
    )

import jax


def _get_ring_info():
    info = get_benchmark("ring")
    if info is None:
        pytest.skip("ring benchmark not found in vajax registry")
    return info


class TestVajaxRing:
    """Run the VAJAX ring oscillator benchmark on Metal."""

    def test_ring_parse(self):
        """Verify the ring benchmark parses correctly."""
        info = _get_ring_info()
        engine = CircuitEngine(info.sim_path)
        engine.parse()

        assert engine.num_nodes > 0, "No nodes parsed"
        assert len(engine.devices) > 0, "No devices parsed"
        # Ring oscillator: 9 inverter stages + vdd + ground + internal nodes
        assert engine.num_nodes >= 10, f"Expected >=10 nodes, got {engine.num_nodes}"

    def test_ring_transient_short(self):
        """Run a short transient (10 timesteps) to verify basic execution."""
        info = _get_ring_info()
        engine = CircuitEngine(info.sim_path)
        engine.parse()

        # Short run: just 10 timesteps to verify it works
        t_stop = info.dt * 10
        engine.prepare(t_stop=t_stop, dt=info.dt, use_sparse=False)
        result = engine.run_transient()

        assert result.num_steps > 0, "No timesteps returned"
        converged = result.stats.get("convergence_rate", 0)
        assert converged > 0.5, f"Poor convergence: {converged * 100:.0f}%"

    def test_ring_transient_full(self):
        """Run the full ring oscillator transient simulation.

        This is a more demanding test that exercises the full simulation
        loop including adaptive timestepping with while_loop.
        """
        info = _get_ring_info()
        engine = CircuitEngine(info.sim_path)
        engine.parse()

        engine.prepare(t_stop=info.t_stop, dt=info.dt, use_sparse=False)
        result = engine.run_transient()

        assert result.num_steps > 100, f"Too few steps: {result.num_steps}"
        converged = result.stats.get("convergence_rate", 0)
        assert converged > 0.9, f"Poor convergence: {converged * 100:.0f}%"


if __name__ == "__main__":
    # Allow running as a script: JAX_PLATFORMS=iree_metal uv run python test_vajax_ring.py
    info = _get_ring_info()
    print(f"Ring benchmark: {info.title}")
    print(f"  dt={info.dt:.2e}, t_stop={info.t_stop:.2e}")

    print("\n=== Parse test ===")
    engine = CircuitEngine(info.sim_path)
    engine.parse()
    print(f"  {engine.num_nodes} nodes, {len(engine.devices)} devices")

    print("\n=== Short transient (10 steps) ===")
    engine2 = CircuitEngine(info.sim_path)
    engine2.parse()
    engine2.prepare(t_stop=info.dt * 10, dt=info.dt, use_sparse=False)
    result = engine2.run_transient()
    print(f"  {result.num_steps} steps, convergence={result.stats.get('convergence_rate', 0):.0%}")

    print("\n=== Full transient ===")
    engine3 = CircuitEngine(info.sim_path)
    engine3.parse()
    engine3.prepare(t_stop=info.t_stop, dt=info.dt, use_sparse=False)
    result = engine3.run_transient()
    print(f"  {result.num_steps} steps, convergence={result.stats.get('convergence_rate', 0):.0%}")

    print("\nAll ring benchmark tests passed!")
