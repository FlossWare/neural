# Experiment 002: Representational capacity

## Hypothesis

A single-layer learner cannot represent XOR, while a one-hidden-layer learner can, without changing the basic deterministic training/evaluation discipline.

The question is deliberately about representational capacity, not about building a reusable neural-network framework.

## Task

Learn XOR:
- [0,0] -> false
- [0,1] -> true
- [1,0] -> true
- [1,1] -> false

The binary examples are used for training. Additional fractional inputs are held out to check whether the learned nonlinear boundary behaves sensibly between the binary examples.

## Comparison

Two disposable learners are implemented in one plain Java source file:

1. Single-layer learner: a perceptron with two inputs.
2. Multilayer learner: two inputs, two hidden tanh units, and one linear output trained with deterministic gradient descent.

Both start from deterministic initial state and use the same examples in the same order.

## Measurements

Record initial training accuracy, final training accuracy, training epochs, held-out accuracy, final state, and state restoration equivalence.

The single-layer learner is expected to plateau below perfect XOR accuracy. The multilayer learner should reach perfect binary training accuracy and classify the fractional held-out examples consistently with XOR.

## Persistence test

Train the multilayer learner, capture its state, construct a fresh learner from that state, and require prediction equivalence on the held-out examples.

## Not part of Experiment 002

- Maven
- frameworks
- external ML libraries
- Loom / loom-ai
- AI teachers
- persistence infrastructure
- generalized neural abstractions
- GPU execution

A disposable experiment has earned exactly zero architecture until the evidence says otherwise.