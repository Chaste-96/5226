# Transport-RL

A reinforcement learning framework for multi-agent transportation problems in grid worlds.

## Project Structure

```
transport‑rl/
│
├─ env/                     # All environment related files
│   ├─ gridworld.py         # GridWorldEnv main class
│   ├─ scheduler.py         # CentralClock / RoundRobin scheduler
│   ├─ sensors.py           # "Reverse occupation" sensors
│   └─ config.py            # Global constants (GRID_SIZE, COSTS...)
│
├─ agents/
│   ├─ q_agent.py           # Tabular Q‑learning version
│   └─ shared_dqn.py        # DQN API wrapper
│
├─ train.py                 # Training entry point (CLI parameters)
├─ evaluate.py              # Performance evaluation script
│
├─ utils/
│   ├─ replay_buffer.py     # Simple wrapper for DQN
│   ├─ logger.py            # Unified printing / CSV saving
│   └─ metrics.py           # Statistics for success rate, collisions, steps
│
├─ tests/                   # Pytest unit tests
│   ├─ test_env.py
│   ├─ test_scheduler.py
│   └─ test_collision.py
│
├─ requirements.txt         # Dependencies (excluding torch)
└─ notebook.ipynb           # Submission notebook importing modules
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Training:
```bash
python train.py
```

Evaluation:
```bash
python evaluate.py
``` 