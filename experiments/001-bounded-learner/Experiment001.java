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
        var updates = learner.train(training);

        check(learner.predict(0, 0) == false, "training example [0,0]");
        check(learner.predict(1, 0), "training example [1,0]");
        check(learner.predict(0, 1), "training example [0,1]");
        check(learner.predict(1, 1), "held-out example [1,1]");

        var restored = new Perceptron(learner.state());
        check(restored.predict(1, 1), "restored held-out prediction");

        System.out.printf("updates=%d%nstate=%s%nheld-out-accuracy=1.0%n", updates, learner);
        System.out.println("Experiment 001 PASSED");
    }

    private static void check(boolean condition, String description) {
        if (!condition) throw new AssertionError(description);
    }
}
