package org.flossware.neural;

import java.util.List;
import java.util.Objects;

public final class Evaluator {
    public double accuracy(Learner learner, List<LabeledExample> examples) {
        Objects.requireNonNull(learner, "learner");
        Objects.requireNonNull(examples, "examples");
        if (examples.isEmpty()) throw new IllegalArgumentException("examples must not be empty");
        long correct = examples.stream()
                .filter(example -> learner.predict(example.features()).label() == example.label())
                .count();
        return (double) correct / examples.size();
    }
}
