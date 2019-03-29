import classifiers.neural.neural_constants as constants
import classifiers.neural.data_slicer as data_slicer
import numpy as np

from classifier_interface import ClassifierWrapper as CW
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from classifiers.neural.glove import load_glove
import csv

class ComparativeNeuralClassifier(CW):
    def __init__(self):
        # a list of tuples of (type, data, true_label)
        self.labelled_data = []
        self.labelled_validation_data = []
        self.labels = set()
        self.models = dict()
        self.tokenizer = None
    #force

    def add_data(self, file_id : str, tokenized_file : str, true_label : int):
        """

		:param file_id: a hashable ID for this particular file
		:param tokenized_file: a
		:param true_label:
		:return: None
		"""

        # CURRENTLY NOT TAKING IN PRE-TOKENIZED FILE, DISCUSS WITH TEAM ABOUT ALTERING CLASSIFIER INTERFACES
        self.labels.add(true_label)
        self.labelled_data.append((file_id, tokenized_file, true_label))

    def add_validation_data(self, file_id : str, data : str, true_label : int):
        """

		:param file_id:
		:param data:
		:param true_label:
		:return:
		"""

        self.labelled_validation_data.append((file_id, data, true_label))

    def get_data(self):
        """

		:return: A structure [(file_id, tokenized_file, true_label),...] for all data added to this classifier with
		the add_data method
		"""
        raise NotImplementedError

    def train(self):
        """
		This classifier object will train on all the data that has been added to it using the adddata method
		:return:
		"""

        # i want to use bagging

        # create the tokenizer
        self.tokenizer = Tokenizer(num_words=constants.MAX_NUMBER_TOKENS)
        training_data = [text for _, text, _ in self.labelled_data]
        self.tokenizer.fit_on_texts(training_data)

        # now build our training data
        X_train = self.tokenizer.texts_to_sequences(training_data)
        X_validation = self.tokenizer.texts_to_sequences([text for _, text, _ in self.labelled_validation_data])

        X_train, y_train = data_slicer.slice_data(X_train,
                                                  [y for _, _, y in self.labelled_data],
                                                  slice_length=constants.SLICE_LENGTH,
                                                  overlap_percent=constants.SLICE_OVERLAP)

        X_validation, y_validation = data_slicer.slice_data(X_validation,
                                                            [y for _, _, y in self.labelled_validation_data],
                                                            slice_length=constants.SLICE_LENGTH,
                                                            overlap_percent=constants.SLICE_OVERLAP)

        # pad them as necessary
        X_train = np.array([np.array(x) for x in pad_sequences(X_train, padding="post", maxlen=constants.SLICE_LENGTH)])
        X_validation = np.array(pad_sequences(X_validation, padding="post", maxlen=constants.SLICE_LENGTH))

        # force change

        # get our glove embeddings
        glove = load_glove(constants.GLOVE_FILE, self.tokenizer.word_index)

        # compute some neural_constants
        vocab_size = len(self.tokenizer.word_index) + 1

        for label in self.labels:
            # set model parameters
            self.models[label] = Sequential()
            model_layers = [
                # must have these two layers firsts
                layers.Embedding(vocab_size,
                                 constants.GLOVE_DIMENSIONS,
                                 weights=[glove],
                                 input_length=constants.SLICE_LENGTH,
                                 trainable=False),
                layers.GlobalMaxPool1D(),
                # now we have some options
                layers.Dense(20, activation="relu"),
                layers.Dense(15, activation="sigmoid"),
                # layers.Dense(10, activation="sigmoid"),
                # probably want a final sigmoid layer to get smooth value in range (0, 1)
                layers.Dense(1, activation="sigmoid")
            ]
            # add them in
            for layer in model_layers:
                self.models[label].add(layer)
            self.models[label].compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            y_train_binary = [1 if l == label else 0 for l in y_train]

            # now we fit (can take a while)
            self.models[label].fit(X_train, y_train_binary,
                           epochs=25,
                           verbose=False,
                           shuffle=True,
                           validation_data=(X_validation, y_validation),
                           batch_size=10)
        predictions = dict()
        stats = dict()
        for label in self.labels:
            predictions[label] = self.models[label].predict(X_validation, verbose=False)
        for label in self.labels:
            stats[label] = { "mean" : np.mean(predictions[label]),
                             "std" : np.std(predictions[label]),
                             "max" : np.max(predictions[label]),
                             "min" : np.min(predictions[label])
                             }

        texts = self.tokenizer.sequences_to_texts(X_validation)

        sorted_labels = sorted(list(self.labels))

        ncorrect = [0] * 4
        n = 0
        with open('classifiers/neural/cnc.csv', 'w', newline='\n') as csvfile:
            csvw = csv.writer(csvfile)
            for i in range(len(y_validation)):

                outputs = [predictions[label][i][0] for label in sorted_labels]
                zscores = [(predictions[label][i][0] - stats[label]["mean"]) / stats[label]["std"] for label in
                           sorted_labels]
                normalized = [(predictions[label][i][0] - stats[label]["min"]) / stats[label]["max"] for label in
                              sorted_labels]
                pred = [np.argmax([(outputs[j] + zscores[j]) for j in range(len(outputs))]),
                        np.argmax(outputs),
                        np.argmax(zscores),
                        np.argmax(normalized)]

                n += 1
                for j in range(len(pred)):
                    if pred[j] == y_validation[j]:
                        ncorrect[j] += 1

                row = [y_validation[i]] \
                      + normalized \
                      + outputs  \
                      + zscores \
                      + pred \
                      + [texts[i]]
                csvw.writerow(row)

        print(ncorrect)
        print([x / n for x in ncorrect])
        print(n)


    def predict(self, tokenized_file : str, minimum_confidence=.8):
        """

		:param tokenized_file: the array containing the ordered, sanitized word tokens from a single file
		:param minimum_confidence: the minimum confidence level required to the classifier to label a data point as
		any given class. Only used by applicable classifiers.
		:return: a list of tuples of [(class label, confidence)] for each class label where confidence >
		minimum_confidence. Confidence will be 1 for classifiers where confidence is not a normally used feature.
		"""

        raise NotImplementedError
