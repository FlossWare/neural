# Experiment 001: Bounded learner

## Hypothesis

A very small learner can acquire a simple binary behavior from examples and generalize beyond the examples used for learning.

The experiment is successful only if the learner behavior can be distinguished from simply remembering the training examples.

## Task

Use a two-feature binary classification problem with a linearly separable rule. For example, label = (x1 + x2) >= 1.

Keep the first task tiny, deterministic, linearly separable, and easy to inspect by hand.

## Learner

Implement the smallest useful trainable unit ourselves.

Required state:
- one weight per feature
- bias
- learning rate

Required operations:
- predict an input
- update state from a labeled example
- expose or capture learned state
- restore learned state

Do not introduce a neural-network framework for this experiment.

## Training

Use a fixed training set containing only part of the possible input space. Run examples in a fixed order so the experiment is reproducible.

Record initial state, state after updates or training passes, number of mistakes, and final state.

## Evaluation

Evaluate against a separate held-out set containing examples not used during training.

Measure training accuracy, held-out accuracy, number of training updates, and passes required for convergence if convergence occurs.

The important observation is whether the learned rule handles unseen examples.

## Persistence test

1. train the learner
2. capture its learned state
3. create a fresh learner from that state
4. evaluate both learners on the same held-out set

The restored learner should produce the same predictions as the original learner.

## Questions this experiment should answer

- What is the minimum state that actually represents what was learned?
- Is the learner learning a rule or merely fitting the supplied examples?
- Does state restoration preserve behavior?
- What observations should become part of a future learner contract?
- Which concepts belong in neural rather than in an execution framework?

## Not part of Experiment 001

- external model APIs
- AI teachers
- embeddings
- GPU acceleration
- distributed execution
- Loom integration
- generalized neural-network frameworks

These are deliberately deferred until the learning behavior gives us a reason to need them.

## Next experiment candidates

Depending on the result: non-linearly separable behavior, noisy examples, incremental learning after persistence, concept drift, multiple cooperating learners, and teacher-generated examples from neural-ai.
