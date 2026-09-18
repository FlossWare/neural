package org.flossware.neural;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

public final class PerceptronStateStore {
    public void save(Perceptron learner, Path path) throws IOException {
        var weights = Arrays.stream(learner.weights())
                .mapToObj(Double::toString)
                .reduce((a, b) -> a + "," + b)
                .orElseThrow();
        Files.writeString(path, weights + System.lineSeparator()
                + learner.bias() + System.lineSeparator()
                + learner.learningRate() + System.lineSeparator());
    }

    public Perceptron load(Path path) throws IOException {
        var lines = Files.readAllLines(path);
        if (lines.size() != 3) throw new IOException("Invalid perceptron state");
        var weights = Arrays.stream(lines.get(0).split(",")).mapToDouble(Double::parseDouble).toArray();
        return new Perceptron(weights, Double.parseDouble(lines.get(1)), Double.parseDouble(lines.get(2)));
    }
}
