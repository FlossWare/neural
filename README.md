# Neural

Experimental neural learning research for FlossWare.

## Purpose

This repository explores small neural learners, learning behavior, persistence, evaluation, lineage, and eventually collective/evolutionary learning.

The project deliberately starts small. It is research code, not a production framework.

## Initial direction

The first experiments will establish whether a small learner can:

1. learn a bounded task from examples,
2. retain what it learned,
3. generalize beyond the examples,
4. persist learned state,
5. participate in repeatable experiments.

AI/model teachers are intentionally kept outside this repository's core concerns. Those experiments belong in `neural-ai`.

## Relationship to other FlossWare projects

- `neural`: neural learning research and core concepts.
- `neural-ai`: neural learning using AI models as teachers or knowledge sources.
- `loom`: language-neutral execution substrate.
- `loom-ai`: AI execution contracts and semantics.

Loom integration is expected later. Early experiments may use disposable implementations where that produces evidence faster.

## Status

Research / experimental.
