# Shadow Expert Planner

A standalone planner for selecting shadow-expert replicas in Mixture-of-Experts
systems. Given route counts for two micro-batches and expert ownership, it returns
replica IDs for each rank and predicted timing metrics.

## Features

- **Topology-independent (`cost`) and NUMA-aware (`numa`) models.**
- **Communication-aware selection:** accounts for compute/communication overlap
  and exposed parameter-transfer and gradient-synchronization costs.
- **Planning and evaluation:** greedy selection with local swaps, plus independent
  evaluation of existing replica plans.
- **Python and JSON interfaces**, with examples using synthetic route counts.

## Quick start

Requires Python 3.9+ and PyTorch 2.0+. Runs on CPU without a distributed setup.
Run these commands from the project root:

```bash
python -m pip install -e .
python -m examples.demo_selection
```

To generate a NUMA-aware plan from JSON:

```bash
python -m shadow_planner examples/sample_counts.json --model numa
```

The package produces plans and timing estimates; replica execution belongs to the
training runtime. Calibrate timing coefficients for your hardware and workload.

## License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
