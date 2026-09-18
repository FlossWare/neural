package flossware.neural;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;

class PerceptronTest {
    private static final List<Perceptron.Example> TRAIN = List.of(
            new Perceptron.Example(new double[]{0, 0}, false),
            new Perceptron.Example(new double[]{1, 0}, true),
            new Perceptron.Example(new double[]{0, 1}, true));

    private static final List<Perceptron.Example> HELD_OUT = List.of(
            new Perceptron.Example(new double[]{1, 1}, true));

    @Test
    void learnsAndGeneralizesToHeldOutExample() {
        var learner = new Perceptron(2, 1.0);
        assertTrue(learner.train(TRAIN) > 0);
        assertTrue(learner.predict(1, 1));
        assertTrue(learner.predict(1, 0));
        assertTrue(learner.predict(0, 1));
        assertFalse(learner.predict(0, 0));
        assertEquals(1.0, accuracy(learner, HELD_OUT));
    }

    @Test
    void capturedStatePreservesBehavior() {
        var learner = new Perceptron(2, 1.0);
        learner.train(TRAIN);
        var state = learner.state();
        assertArrayEquals(learner.state().weights(), state.weights());
        assertEquals(learner.state().bias(), state.bias());
    }

    private static double accuracy(Perceptron learner, List<Perceptron.Example> examples) {
        return examples.stream().filter(e -> learner.predict(e.features()) == e.label()).count() / (double) examples.size();
    }
}
