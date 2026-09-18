package org.flossware.neural;

public record PerceptronState(double[] weights, double bias, double learningRate) {
    public PerceptronState {
        if (weights == null || weights.length == 0) throw new IllegalArgumentException("weights must not be empty");
        if (!(learningRate > 0.0)) throw new IllegalArgumentException("learningRate must be positive");
        weights = weights.clone();
    }

    @Override
    public double[] weights() {
        return weights.clone();
    }
}
