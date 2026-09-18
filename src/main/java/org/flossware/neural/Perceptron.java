package org.flossware.neural;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class Perceptron implements Learner {
    private final double learningRate;
    private double[] weights;
    private double bias;

    public Perceptron(int featureCount, double learningRate) {
        if (featureCount < 1) throw new IllegalArgumentException("featureCount must be positive");
        if (!(learningRate > 0.0)) throw new IllegalArgumentException("learningRate must be positive");
        this.weights = new double[featureCount];
        this.learningRate = learningRate;
    }

    public Perceptron(double[] weights, double bias, double learningRate) {
        Objects.requireNonNull(weights, "weights");
        if (weights.length == 0) throw new IllegalArgumentException("weights must not be empty");
        if (!(learningRate > 0.0)) throw new IllegalArgumentException("learningRate must be positive");
        this.weights = weights.clone();
        this.bias = bias;
        this.learningRate = learningRate;
    }

    @Override
    public void train(List<LabeledExample> examples) {
        Objects.requireNonNull(examples, "examples");
        for (var example : examples) {
            var x = example.features().values();
            if (x.length != weights.length) throw new IllegalArgumentException("feature count changed");
            var expected = example.label() ? 1.0 : 0.0;
            var actual = predict(example.features()).label() ? 1.0 : 0.0;
            var error = expected - actual;
            if (error != 0.0) {
                for (int i = 0; i < weights.length; i++) {
                    weights[i] += learningRate * error * x[i];
                }
                bias += learningRate * error;
            }
        }
    }

    @Override
    public Prediction predict(FeatureVector features) {
        var x = features.values();
        if (x.length != weights.length) throw new IllegalArgumentException("feature count changed");
        var activation = bias;
        for (int i = 0; i < weights.length; i++) activation += weights[i] * x[i];
        return new Prediction(activation >= 0.0, activation);
    }

    public double[] weights() {
        return weights.clone();
    }

    public double bias() {
        return bias;
    }

    public double learningRate() {
        return learningRate;
    }

    @Override
    public String toString() {
        return "Perceptron{weights=" + Arrays.toString(weights) + ", bias=" + bias + ", learningRate=" + learningRate + '}';
    }
}
