# Triplet STDP CV2 MNIST Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch MNIST experiment that compares the average angle between standard BP and proposed biological-rule weight-change vectors across epochs.

**Architecture:** Keep the neural model, learning rules, and experiment runner in separate modules. The model exposes cached pre-activations and positive activations; the learning-rule module computes candidate update vectors; the runner records CSV metrics and plots the trend.

**Tech Stack:** Python, PyTorch, torchvision, matplotlib, pytest.

---

### Task 1: Learning-Rule Tests

**Files:**
- Create: `tests/test_learning_rules.py`

- [x] Add tests for zero-vector angle handling, right-angle computation, and matching BP/biological delta shapes.
- [x] Attempt to run pytest and capture environment limitations.

### Task 2: Model And Learning Rules

**Files:**
- Create: `triplet_stdp_cv2_learning/model.py`
- Create: `triplet_stdp_cv2_learning/learning_rules.py`
- Create: `triplet_stdp_cv2_learning/__init__.py`

- [x] Implement `PositiveMLP` with `softplus(x) + eps` positive activations.
- [x] Implement BP deltas from autograd.
- [x] Implement biological error recurrence, zeta scaling, update-function variants, and angle statistics.

### Task 3: MNIST Experiment Runner

**Files:**
- Create: `triplet_stdp_cv2_learning/train_mnist.py`
- Create: `requirements.txt`

- [x] Implement MNIST loading with torchvision.
- [x] Implement epoch loop that computes BP and biological update vectors on the same batch before applying the selected training rule.
- [x] Save per-epoch metrics to CSV and plot the angle trend.

### Task 4: Documentation And Verification

**Files:**
- Create: `README.md`
- Create: `docs/algorithm_details.md`
- Create: `.gitignore`

- [x] Document installation, commands, parameters, outputs, and formula-to-code mapping.
- [x] Run `python3 -m compileall triplet_stdp_cv2_learning tests`.
- [x] Note that full pytest/training requires installing PyTorch, torchvision, matplotlib, and pytest.
