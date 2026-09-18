package flossware.neural;

import java.util.List;

public final class Experiment001 {
    private static final int MAX_PASSES = 100;

    public static void main(String[] args) {
        var training = List.of(
                new Perceptron.Example(new double[]{1, 0}, true),
                new Perceptron.Example(new double[]{0, 1}, true),
                new Perceptron.Example(new double[]{0, 0}, false));

        var heldOut = List.of(
                new Perceptron.Example(new double[]{1, 1}, true));

        var learner = new Perceptron(2, 1.0);
        var initialState = learner.state();

        int updates = 0;
        int passes = 0;
        while (passes < MAX_PASSES && countCorrect(learner, training) < training.size()) {
            updates += learner.train(training);
            passes++;
        }

        var finalState = learner.state();
        var trainingCorrect = countCorrect(learner, training);
        var heldOutCorrect = countCorrect(learner, heldOut);

        check(trainingCorrect == training.size(), "training accuracy");
        check(heldOutCorrect == heldOut.size(), "held-out accuracy");
        check(passes > 0, "convergence requires training passes");
        check(passes <= MAX_PASSES, "convergence limit");

        var restored = new Perceptron(finalState);
        check(countCorrect(restored, heldOut) == heldOut.size(), "restored held-out accuracy");

        System.out.printf(
                "initial-state=%s%nupdates=%d%ntraining-accuracy=%d/%d%nheld-out-accuracy=%d/%d%npasses=%d%nfinal-state=%s%n",
                initialState,
                updates,
                trainingCorrect, training.size(),
                heldOutCorrect, heldOut.size(),
                passes,
                finalState);
        System.out.println("Experiment 001 PASSED");
    }

    private static int countCorrect(Perceptron learner, List<Perceptron.Example> examples) {
        int correct = 0;
        for (var example : examples) {
            if (learner.predict(example.features()) == example.label()) correct++;
        }
        return correct;
    }

    private static void check(boolean condition, String description) {
        if (!condition) throw new AssertionError(description);
    }
}
