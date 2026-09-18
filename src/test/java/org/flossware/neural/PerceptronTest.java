package org.flossware.neural;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import org.junit.jupiter.api.Test;

class PerceptronTest {
    @Test
    void learnsOrGate() {
        var learner = new Perceptron(2, 0.2);
        var examples = java.util.List.of(
                new LabeledExample(new FeatureVector(new double[]{0, 0}), false),
                new LabeledExample(new FeatureVector(new double[]{0, 1}), true),
                new LabeledExample(new FeatureVector(new double[]{1, 0}), true),
                new LabeledExample(new FeatureVector(new double[]{1, 1}), true));
        for (int i = 0; i < 10; i++) learner.train(examples);
        assertEquals(1.0, new Evaluator().accuracy(learner, examples));
    }

    @Test
    void persistsLearnedState() throws Exception {
        var learner = new Perceptron(2, 0.2);
        var examples = java.util.List.of(
                new LabeledExample(new FeatureVector(new double[]{0, 0}), false),
                new LabeledExample(new FeatureVector(new double[]{1, 0}), true));
        for (int i = 0; i < 10; i++) learner.train(examples);

        var path = Files.createTempFile("perceptron-", ".state");
        new PerceptronStateStore().save(learner, path);
        var restored = new PerceptronStateStore().load(path);
        assertArrayEquals(learner.weights(), restored.weights());
        assertEquals(learner.bias(), restored.bias());
        assertEquals(learner.learningRate(), restored.learningRate());
        assertEquals(learner.predict(new FeatureVector(new double[]{1, 0})),
                restored.predict(new FeatureVector(new double[]{1, 0})));
    }
}
