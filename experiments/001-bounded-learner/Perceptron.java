package flossware.neural;

import java.util.Arrays;
import java.util.List;

public final class Perceptron {
    private final double learningRate;
    private final double[] weights;
    private double bias;

    public Perceptron(int featureCount, double learningRate) {
        if (featureCount < 1 || !(learningRate > 0.0)) {
            throw new IllegalArgumentException("featureCount and learningRate must be positive");
        }
        this.weights = new double[featureCount];
        this.learningRate = learningRate;
    }

    public boolean predict(double... features) {
        return activation(features) >= 0.0;
    }

    public double activation(double... features) {
        if (features.length != weights.length) throw new IllegalArgumentException("feature count changed");
        double sum = bias;
        for (int i = 0; i < weights.length; i++) sum += weights[i] * features[i];
        return sum;
    }

    public int train(List<Example> examples) {
        int updates = 0;
        for (Example example : examples) {
            boolean actual = predict(example.features());
            if (actual != example.label()) {
                double error = example.label() ? 1.0 : -1.0;
                double[] x = example.features();
                for (int i = 0; i < weights.length; i++) weights[i] += learningRate * error * x[i];
                bias += learningRate * error;
                updates++;
            }
        }
        return updates;
    }

    public State state() {
        return new State(weights.clone(), bias, learningRate);
    }

    public record Example(double[] features, boolean label) {
        public Example {
            if (features == null || features.length == 0) throw new IllegalArgumentException("features required");
            features = features.clone();
        }
        @Override public double[] features() { return features.clone(); }
    }

    public record State(double[] weights, double bias, double learningRate) {
        public State {
            if (weights == null || weights.length == 0) throw new IllegalArgumentException("weights required");
            weights = weights.clone();
        }
        @Override public double[] weights() { return weights.clone(); }
    }

    @Override public String toString() {
        return "Perceptron{weights=" + Arrays.toString(weights) + ", bias=" + bias + ", learningRate=" + learningRate + "}";
    }
}
