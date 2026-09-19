package flossware.neural;

import java.util.List;
import java.util.Random;

public final class Experiment002 {
    private static final int MAX_EPOCHS = 10_000;
    private static final double LEARNING_RATE = 0.1;

    public static void main(String[] args) {
        var training = List.of(
                new Example(new double[]{0, 0}, false),
                new Example(new double[]{0, 1}, true),
                new Example(new double[]{1, 0}, true),
                new Example(new double[]{1, 1}, false));
        var heldOut = List.of(
                new Example(new double[]{0.1, 0.9}, true),
                new Example(new double[]{0.9, 0.9}, false),
                new Example(new double[]{0.2, 0.2}, false),
                new Example(new double[]{0.8, 0.2}, true));

        var single = new Perceptron(2, LEARNING_RATE);
        int singleEpochs = train(single, training);
        int singleTrainingCorrect = countCorrect(single, training);

        var multilayer = new Multilayer(2, 2, LEARNING_RATE, 42L);
        int multilayerEpochs = train(multilayer, training);
        var finalState = multilayer.state();
        int multilayerTrainingCorrect = countCorrect(multilayer, training);
        int heldOutCorrect = countCorrect(multilayer, heldOut);

        check(singleTrainingCorrect < training.size(), "single-layer learner must not perfectly represent XOR");
        check(multilayerTrainingCorrect == training.size(), "multilayer learner must represent XOR");
        check(heldOutCorrect == heldOut.size(), "multilayer held-out accuracy");
        check(multilayerEpochs > 0 && multilayerEpochs <= MAX_EPOCHS, "multilayer convergence bound");

        var restored = new Multilayer(finalState);
        check(predictionsEqual(multilayer, restored, heldOut), "restored multilayer predictions");

        System.out.printf("single-training-accuracy=%d/%d%n"
                + "single-epochs=%d%n"
                + "multilayer-training-accuracy=%d/%d%n"
                + "multilayer-held-out-accuracy=%d/%d%n"
                + "multilayer-epochs=%d%n"
                + "multilayer-final-state=%s%n",
                singleTrainingCorrect, training.size(), singleEpochs,
                multilayerTrainingCorrect, training.size(),
                heldOutCorrect, heldOut.size(), multilayerEpochs, finalState);
        System.out.println("Experiment 002 PASSED");
    }

    private static int train(Perceptron learner, List<Example> examples) {
        int epochs = 0;
        while (epochs < MAX_EPOCHS && countCorrect(learner, examples) < examples.size()) {
            learner.train(examples);
            epochs++;
        }
        return epochs;
    }

    private static int train(Multilayer learner, List<Example> examples) {
        int epochs = 0;
        while (epochs < MAX_EPOCHS && countCorrect(learner, examples) < examples.size()) {
            learner.train(examples);
            epochs++;
        }
        return epochs;
    }

    private static int countCorrect(Perceptron learner, List<Example> examples) {
        int correct = 0;
        for (var example : examples) if (learner.predict(example.features()) == example.label()) correct++;
        return correct;
    }

    private static int countCorrect(Multilayer learner, List<Example> examples) {
        int correct = 0;
        for (var example : examples) if (learner.predict(example.features()) == example.label()) correct++;
        return correct;
    }

    private static boolean predictionsEqual(Multilayer a, Multilayer b, List<Example> examples) {
        for (var example : examples) if (a.predict(example.features()) != b.predict(example.features())) return false;
        return true;
    }

    private record Example(double[] features, boolean label) {
        private Example { features = features.clone(); }
        @Override public double[] features() { return features.clone(); }
    }

    private static final class Perceptron {
        private final double[] weights;
        private final double learningRate;
        private double bias;
        private Perceptron(int featureCount, double learningRate) { this.weights = new double[featureCount]; this.learningRate = learningRate; }
        private boolean predict(double... x) {
            double sum = bias;
            for (int i = 0; i < weights.length; i++) sum += weights[i] * x[i];
            return sum >= 0.0;
        }
        private void train(List<Example> examples) {
            for (var e : examples) if (predict(e.features()) != e.label()) {
                double error = e.label() ? 1.0 : -1.0;
                double[] x = e.features();
                for (int i = 0; i < weights.length; i++) weights[i] += learningRate * error * x[i];
                bias += learningRate * error;
            }
        }
    }

    private static final class Multilayer {
        private final double[][] hiddenWeights;
        private final double[] hiddenBiases;
        private final double[] outputWeights;
        private final double learningRate;
        private double outputBias;

        private Multilayer(int inputs, int hidden, double learningRate, long seed) {
            hiddenWeights = new double[hidden][inputs]; hiddenBiases = new double[hidden];
            outputWeights = new double[hidden]; this.learningRate = learningRate;
            var random = new Random(seed);
            for (int h = 0; h < hidden; h++) {
                for (int i = 0; i < inputs; i++) hiddenWeights[h][i] = random.nextDouble() * 2.0 - 1.0;
                hiddenBiases[h] = random.nextDouble() * 2.0 - 1.0;
                outputWeights[h] = random.nextDouble() * 2.0 - 1.0;
            }
            outputBias = random.nextDouble() * 2.0 - 1.0;
        }

        private Multilayer(State s) {
            hiddenWeights = copy(s.hiddenWeights()); hiddenBiases = s.hiddenBiases().clone();
            outputWeights = s.outputWeights().clone(); outputBias = s.outputBias(); learningRate = s.learningRate();
        }

        private boolean predict(double... x) { return output(x) >= 0.0; }

        private double output(double... x) {
            double[] h = hidden(x); double result = outputBias;
            for (int j = 0; j < outputWeights.length; j++) result += outputWeights[j] * h[j];
            return result;
        }

        private void train(List<Example> examples) {
            for (var e : examples) {
                double[] x = e.features(); double[] h = hidden(x);
                double target = e.label() ? 1.0 : -1.0; double error = output(x) - target;
                double[] old = outputWeights.clone();
                for (int j = 0; j < outputWeights.length; j++) outputWeights[j] -= learningRate * error * h[j];
                outputBias -= learningRate * error;
                for (int j = 0; j < hiddenWeights.length; j++) {
                    double delta = error * old[j] * (1.0 - h[j] * h[j]);
                    for (int i = 0; i < hiddenWeights[j].length; i++) hiddenWeights[j][i] -= learningRate * delta * x[i];
                    hiddenBiases[j] -= learningRate * delta;
                }
            }
        }

        private double[] hidden(double[] x) {
            double[] h = new double[hiddenWeights.length];
            for (int j = 0; j < hiddenWeights.length; j++) {
                double sum = hiddenBiases[j];
                for (int i = 0; i < hiddenWeights[j].length; i++) sum += hiddenWeights[j][i] * x[i];
                h[j] = Math.tanh(sum);
            }
            return h;
        }

        private State state() { return new State(copy(hiddenWeights), hiddenBiases.clone(), outputWeights.clone(), outputBias, learningRate); }
        private record State(double[][] hiddenWeights, double[] hiddenBiases, double[] outputWeights, double outputBias, double learningRate) {}
        private static double[][] copy(double[][] source) {
            double[][] result = new double[source.length][];
            for (int i = 0; i < source.length; i++) result[i] = source[i].clone();
            return result;
        }
    }

    private static void check(boolean condition, String description) { if (!condition) throw new AssertionError(description); }
}
