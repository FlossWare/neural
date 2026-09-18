package org.flossware.neural;

import java.util.Objects;

public record LabeledExample(FeatureVector features, boolean label) {
    public LabeledExample {
        Objects.requireNonNull(features, "features");
    }
}
