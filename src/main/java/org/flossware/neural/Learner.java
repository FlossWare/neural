package org.flossware.neural;

import java.util.List;

public interface Learner {
    void train(List<LabeledExample> examples);

    Prediction predict(FeatureVector features);
}
