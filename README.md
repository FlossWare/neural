# neural

Experimental neural learning research for FlossWare.

This repository explores small, understandable learning systems before introducing framework or infrastructure complexity.

## Research direction

The initial question is deliberately small:

> Can a simple learner acquire a bounded behavior from examples, retain what it learned, and generalize to examples it did not see?

The experiments will examine:

- learning versus memorization
- retained learner state
- generalization
- deterministic evaluation
- persistence and restoration
- increasingly complex learning rules
- interaction among multiple learners

Core learning mechanisms are intentionally implemented from first principles where practical. External libraries should earn their place by enabling an experiment, not by defining the experiment.

## Relationship to other FlossWare projects

- neural: learning mechanisms and learner behavior.
- neural-ai: experiments where AI models act as teachers or knowledge sources.
- loom: language-neutral execution substrate, introduced when experiments need it.
- loom-ai: AI execution contracts and model-provider semantics.

neural does not depend on neural-ai. A future integration should pass teaching signals into a learner through a small, language-neutral boundary.

## Current experiment

Experiment 001 defines the first research target: a minimal binary learner evaluated on held-out examples.

No Maven project, framework, or Loom dependency is required at this stage. The point is to learn what the learner needs before building a small software cathedral around it.

## Research discipline

Each experiment should state:

1. hypothesis
2. smallest implementation needed
3. training data
4. evaluation data
5. measurements
6. observed result
7. what the result changes about the next experiment

Results should be reproducible where practical, and negative results are useful evidence rather than failures to hide.
