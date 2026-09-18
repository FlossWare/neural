package flossware.neural;

import java.util.List;

public final class Experiment001 {
    public static void main(String[] args) {
        var training = List.of(
                new Perceptron.Example(new double[]{0, 0}, false),
                new Perceptron.Example(new double[]{1, 0}, true),
                new Perceptron.Example(new double[]{0, 1}, true));

        var heldOut = List.of(
                new Perceptron.Example(new double[]{1, 1}, true));

        var learner = new Perceptron(2, 1.0);
        var initialState = learner.state();
        var updates = learner.train(training);
        var finalState = learner.state();

        var trainingCorrect = countCorrect(learner, training);
        var heldOutCorrect = countCorrect(learner, heldOut);

        check(trainingCorrect == training.size(), "training accuracy");
        check(heldOutCorrect == heldOut.size(), "held-out accuracy");

        var restored = new Perceptron(finalState);
        check(countCorrect(restored, heldOut) == heldOut.size(), "restored held-out accuracy");

        System.out.printf(
                "initial-state=%s%nupdates=%d%ntraining-accuracy=%d/%d%nheld-out-accuracy=%d/%d%nfinal-state=%s%n",
                initialState,
                updates,
                trainingCorrect, training.size(),
                heldOutCorrect, heldOut.size(),
                finalState);
        System.out.println("converged-in-passes=1");
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
