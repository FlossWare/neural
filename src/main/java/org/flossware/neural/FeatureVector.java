package org.flossware.neural;

import java.util.Arrays;
import java.util.Objects;

public record FeatureVector(double[] values) {
    public FeatureVector {
        Objects.requireNonNull(values, "values");
        if (values.length == 0) {
            throw new IllegalArgumentException("A feature vector must contain at least one value");
        }
        values = values.clone();
    }

    @Override
    public double[] values() {
        return values.clone();
    }

    public int size() {
        return values.length;
    }

    @Override
    public String toString() {
        return Arrays.toString(values);
    }
}
