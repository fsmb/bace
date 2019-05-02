# base class for all classifiers
from typing import List, Text, Tuple

class Classifier:
    def read_data(filename: Text) -> Tuple[List[Text], List[Text]]:
        raise NotImplementedError()

    def get_classifier(train_labels: List[Text], train_data:List[Text], num: int = 500):
        raise NotImplementedError()

    def predict(classifier, test_data:List[Text], num: int = 200):
        raise NotImplementedError()