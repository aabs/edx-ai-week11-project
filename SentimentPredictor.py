from sklearn.linear_model import SGDClassifier


class SentimentPredictor:
    def __init__(self):
        self.reporter = None
        self.test_labels = None
        self.training_labels = None
        self.test_data = None
        self.training_data = None
        self.cls = SGDClassifier()

    def with_training_data(self, td):
        return self

    def with_training_labels(self, tl):
        return self

    def with_test_data(self, td):
        return self

    def with_test_labels(self, tl):
        return self

    def build(self) -> SentimentPredictorImpl:
        return SentimentPredictorImpl(
            self.training_data,
            self.test_data,
            self.training_labels,
            self.test_labels,
            self.reporter,
            self.cls)

    def with_reporter(self, reporter):
        self.reporter = reporter
        return self


def to_vector(text):
    pass


class SentimentPredictorImpl:
    def __init__(self, training_data, test_data, training_labels, test_labels, reporter, classifier:SGDClassifier):
        self.classifier = classifier
        self.reporter = reporter
        self.test_labels = test_labels
        self.training_labels = training_labels
        self.test_data = test_data
        self.training_data = training_data

    def fit(self):
        self.classifier.fit(self.training_data, self.training_labels)

    def predict(self, text):
        v = to_vector(text)
        self.classifier.predic(v)