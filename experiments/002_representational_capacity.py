#!/usr/bin/env python3
"""Experiment 002: test representational capacity with a tiny multilayer learner.

This is deliberately self-contained research code. It has no framework or
third-party dependency so the learning behavior remains visible.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path


TRAINING = (
    ((-1.0, -1.0), 0),
    ((-1.0, 1.0), 1),
    ((1.0, -1.0), 1),
    ((1.0, 1.0), 0),
)

HELD_OUT = (
    ((-0.5, -0.5), 0),
    ((-0.5, 0.5), 1),
    ((0.5, -0.5), 1),
    ((0.5, 0.5), 0),
)

SEED = 7
LEARNING_RATE = 0.5
EPOCHS = 10_000
CHECKPOINT_EPOCH = 1_000


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class SingleLayerLearner:
    """A deterministic binary linear learner used as the baseline."""

    def __init__(self) -> None:
        self.weights = [0.0, 0.0]
        self.bias = 0.0

    def predict(self, features: tuple[float, float]) -> int:
        score = sum(w * x for w, x in zip(self.weights, features)) + self.bias
        return int(score >= 0.0)

    def train(self, examples: tuple[tuple[tuple[float, float], int], ...]) -> None:
        for _ in range(50):
            for features, expected in examples:
                error = expected - self.predict(features)
                self.weights[0] += 0.2 * error * features[0]
                self.weights[1] += 0.2 * error * features[1]
                self.bias += 0.2 * error

    def state(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}

    @classmethod
    def restore(cls, state: dict) -> "SingleLayerLearner":
        learner = cls()
        learner.weights = list(state["weights"])
        learner.bias = state["bias"]
        return learner


class TinyMultilayerLearner:
    """A one-hidden-layer sigmoid learner with explicit state."""

    def __init__(self, hidden_size: int = 4, seed: int = SEED) -> None:
        rng = random.Random(seed)
        self.weights_1 = [
            [rng.uniform(-1.0, 1.0) for _ in range(2)] for _ in range(hidden_size)
        ]
        self.bias_1 = [rng.uniform(-1.0, 1.0) for _ in range(hidden_size)]
        self.weights_2 = [rng.uniform(-1.0, 1.0) for _ in range(hidden_size)]
        self.bias_2 = rng.uniform(-1.0, 1.0)

    def _forward(self, features: tuple[float, float]) -> tuple[float, list[float]]:
        hidden = [
            sigmoid(
                self.weights_1[j][0] * features[0]
                + self.weights_1[j][1] * features[1]
                + self.bias_1[j]
            )
            for j in range(len(self.weights_1))
        ]
        output = sigmoid(
            sum(w * h for w, h in zip(self.weights_2, hidden)) + self.bias_2
        )
        return output, hidden

    def predict(self, features: tuple[float, float]) -> int:
        output, _ = self._forward(features)
        return int(output >= 0.5)

    def train_epoch(
        self, examples: tuple[tuple[tuple[float, float], int], ...]
    ) -> None:
        for features, expected in examples:
            output, hidden = self._forward(features)
            output_delta = (output - expected) * output * (1.0 - output)
            old_output_weights = self.weights_2[:]

            for j in range(len(self.weights_2)):
                self.weights_2[j] -= LEARNING_RATE * output_delta * hidden[j]
            self.bias_2 -= LEARNING_RATE * output_delta

            for j in range(len(self.weights_1)):
                hidden_delta = (
                    output_delta
                    * old_output_weights[j]
                    * hidden[j]
                    * (1.0 - hidden[j])
                )
                self.weights_1[j][0] -= (
                    LEARNING_RATE * hidden_delta * features[0]
                )
                self.weights_1[j][1] -= (
                    LEARNING_RATE * hidden_delta * features[1]
                )
                self.bias_1[j] -= LEARNING_RATE * hidden_delta

    def state(self) -> dict:
        return {
            "weights_1": self.weights_1,
            "bias_1": self.bias_1,
            "weights_2": self.weights_2,
            "bias_2": self.bias_2,
        }

    @classmethod
    def restore(cls, state: dict) -> "TinyMultilayerLearner":
        learner = cls(hidden_size=len(state["weights_1"]))
        learner.weights_1 = [list(row) for row in state["weights_1"]]
        learner.bias_1 = list(state["bias_1"])
        learner.weights_2 = list(state["weights_2"])
        learner.bias_2 = state["bias_2"]
        return learner


def evaluate(learner, examples) -> dict:
    mistakes = [
        {"features": list(features), "expected": expected, "actual": learner.predict(features)}
        for features, expected in examples
        if learner.predict(features) != expected
    ]
    return {
        "accuracy": (len(examples) - len(mistakes)) / len(examples),
        "mistakes": mistakes,
    }


def main() -> None:
    baseline = SingleLayerLearner()
    baseline.train(TRAINING)

    learner = TinyMultilayerLearner()
    initial = evaluate(learner, TRAINING)
    initial_held_out = evaluate(learner, HELD_OUT)

    checkpoints = {0: {"training": initial, "held_out": initial_held_out}}

    for epoch in range(1, EPOCHS + 1):
        learner.train_epoch(TRAINING)
        if epoch == CHECKPOINT_EPOCH:
            checkpoints[epoch] = {
                "training": evaluate(learner, TRAINING),
                "held_out": evaluate(learner, HELD_OUT),
            }

    final_training = evaluate(learner, TRAINING)
    final_held_out = evaluate(learner, HELD_OUT)

    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "learner.json"
        state_path.write_text(json.dumps(learner.state(), sort_keys=True), encoding="utf-8")
        restored = TinyMultilayerLearner.restore(
            json.loads(state_path.read_text(encoding="utf-8"))
        )

    restoration_equivalent = all(
        restored.predict(features) == learner.predict(features)
        for features, _ in TRAINING + HELD_OUT
    )

    result = {
        "experiment": "002-representational-capacity",
        "seed": SEED,
        "epochs": EPOCHS,
        "training_examples": len(TRAINING),
        "held_out_examples": len(HELD_OUT),
        "baseline_single_layer": evaluate(baseline, HELD_OUT),
        "checkpoints": checkpoints,
        "final_training": final_training,
        "final_held_out": final_held_out,
        "state_restoration_equivalent": restoration_equivalent,
        "hypothesis_supported": (
            final_training["accuracy"] == 1.0
            and final_held_out["accuracy"] >= 0.75
            and evaluate(baseline, HELD_OUT)["accuracy"] < final_held_out["accuracy"]
        ),
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
