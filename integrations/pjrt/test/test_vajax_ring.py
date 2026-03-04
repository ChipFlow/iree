"""Test VAJAX ring oscillator benchmark on Metal via IREE PJRT.

Requires: vajax optional dependency (uv sync --extra vajax)
Run: JAX_PLATFORMS=cpu,iree_metal uv run python -m pytest test_vajax_ring.py -v

Note: cpu platform is needed because vajax uses jax.devices("cpu") internally
for early collapse decisions during parsing.
"""

import platform
from pathlib import Path

import pytest

# Skip on non-macOS
if platform.system() != "Darwin":
    pytest.skip("Metal requires macOS", allow_module_level=True)

try:
    from vajax.analysis import CircuitEngine
except ImportError:
    pytest.skip(
        "vajax not installed (run: uv sync --extra vajax)", allow_module_level=True
    )

# Ring benchmark parameters (from runme.sim: step=0.05n stop=1u)
RING_DT = 5e-11
RING_T_STOP = 1e-6

# Local fixture path (bundled from VACASK benchmark suite)
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ring" / "vacask"


def _get_ring_sim_path() -> Path:
    """Get ring benchmark sim path: try registry first, fall back to local fixture."""
    try:
        from vajax.benchmarks.registry import get_benchmark

        info = get_benchmark("ring")
        if info is not None:
            return info.sim_path
    except Exception:
        pass

    sim_path = _FIXTURE_DIR / "runme.sim"
    if not sim_path.exists():
        pytest.skip("ring benchmark data not found (neither registry nor fixtures)")
    return sim_path


class TestVajaxRing:
    """Run the VAJAX ring oscillator benchmark on Metal."""

    def test_ring_parse(self):
        """Verify the ring benchmark parses correctly."""
        sim_path = _get_ring_sim_path()
        engine = CircuitEngine(sim_path)
        engine.parse()

        assert engine.num_nodes > 0, "No nodes parsed"
        assert len(engine.devices) > 0, "No devices parsed"
        # Ring oscillator: 9 inverter stages + vdd + ground + internal nodes
        assert engine.num_nodes >= 10, f"Expected >=10 nodes, got {engine.num_nodes}"

    def test_ring_transient_short(self):
        """Run a short transient (10 timesteps) to verify basic execution."""
        sim_path = _get_ring_sim_path()
        engine = CircuitEngine(sim_path)
        engine.parse()

        # Short run: just 10 timesteps to verify it works
        t_stop = RING_DT * 10
        engine.prepare(t_stop=t_stop, dt=RING_DT, use_sparse=False)
        result = engine.run_transient()

        assert result.num_steps > 0, "No timesteps returned"
        converged = result.stats.get("convergence_rate", 0)
        assert converged > 0.5, f"Poor convergence: {converged * 100:.0f}%"

    def test_ring_transient_full(self):
        """Run the full ring oscillator transient simulation.

        This is a more demanding test that exercises the full simulation
        loop including adaptive timestepping with while_loop.
        """
        sim_path = _get_ring_sim_path()
        engine = CircuitEngine(sim_path)
        engine.parse()

        engine.prepare(t_stop=RING_T_STOP, dt=RING_DT, use_sparse=False)
        result = engine.run_transient()

        assert result.num_steps > 100, f"Too few steps: {result.num_steps}"
        converged = result.stats.get("convergence_rate", 0)
        assert converged > 0.9, f"Poor convergence: {converged * 100:.0f}%"


if __name__ == "__main__":
    # Allow running as a script: JAX_PLATFORMS=iree_metal uv run python test_vajax_ring.py
    sim_path = _get_ring_sim_path()
    print(f"Ring benchmark: {sim_path}")
    print(f"  dt={RING_DT:.2e}, t_stop={RING_T_STOP:.2e}")

    print("\n=== Parse test ===")
    engine = CircuitEngine(sim_path)
    engine.parse()
    print(f"  {engine.num_nodes} nodes, {len(engine.devices)} devices")

    print("\n=== Short transient (10 steps) ===")
    engine2 = CircuitEngine(sim_path)
    engine2.parse()
    engine2.prepare(t_stop=RING_DT * 10, dt=RING_DT, use_sparse=False)
    result = engine2.run_transient()
    print(f"  {result.num_steps} steps, convergence={result.stats.get('convergence_rate', 0):.0%}")

    print("\n=== Full transient ===")
    engine3 = CircuitEngine(sim_path)
    engine3.parse()
    engine3.prepare(t_stop=RING_T_STOP, dt=RING_DT, use_sparse=False)
    result = engine3.run_transient()
    print(f"  {result.num_steps} steps, convergence={result.stats.get('convergence_rate', 0):.0%}")

    print("\nAll ring benchmark tests passed!")
