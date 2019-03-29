import classifiers.neural.neural_constants as neural_constants
import classifiers.neural.data_slicer as data_slicer

import numpy as np

from sklearn.metrics import confusion_matrix
import classifiers.neural.neural_constants as neural_constants
import classifiers.neural.data_slicer as data_slicer

from classifier_interface import ClassifierWrapper as CW
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from classifiers.neural.glove import load_glove

from typing import List, Dict

def load_glove(fname : str, word_index : Dict[str, int]):
    """
	Loads the dictionary of glove_embeddings vectors from a given input file

	:param f: a Glove file, open for reading
	:return: a dict of {word : vector, ...} for each word line in the file, where vector is a float array
	"""

    with open(fname , encoding="utf8") as f:
        data = np.zeros((len(word_index) + 1, neural_constants.GLOVE_DIMENSIONS))
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                data[word_index[word]] = np.array(vector, dtype=np.float32)
        return data


class NeuralClassifier(CW):

    def __init__(self):
        # a list of tuples of (type, data, true_label)
        self.labelled_data = []
        self.labelled_validation_data = []
        self.model = None
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
        self.tokenizer = Tokenizer(num_words=neural_constants.MAX_NUMBER_TOKENS)
        # create the tokenizer
        self.tokenizer = Tokenizer(nb_words=neural_constants.MAX_NUMBER_TOKENS)
        training_data = [text for _, text, _ in self.labelled_data]
        self.tokenizer.fit_on_texts(training_data)

        # now build our training data
        X_train = self.tokenizer.texts_to_sequences(training_data)
        X_validation = self.tokenizer.texts_to_sequences([text for _, text, _ in self.labelled_validation_data])

        X_train, y_train = data_slicer.slice_data(X_train,
                                                  [y for _, _, y in self.labelled_data],
                                                  slice_length=neural_constants.SLICE_LENGTH,
                                                  overlap_percent=neural_constants.SLICE_OVERLAP)

        X_validation, y_validation = data_slicer.slice_data(X_validation,
                                                            [y for _, _, y in self.labelled_validation_data],
                                                            slice_length=neural_constants.SLICE_LENGTH,
                                                            overlap_percent=neural_constants.SLICE_OVERLAP)

        # pad them as necessary
        X_train = np.array([np.array(x) for x in pad_sequences(X_train, padding="post", maxlen=neural_constants.SLICE_LENGTH)])
        X_validation = np.array(pad_sequences(X_validation, padding="post", maxlen=neural_constants.SLICE_LENGTH))
        X_train = pad_sequences(X_train, padding="post", maxlen=neural_constants.SLICE_LENGTH)
        X_train = pad_sequences(X_train, padding="post", maxlen=neural_constants.SLICE_LENGTH)

        # force change

        # get our glove embeddings
        glove = load_glove(neural_constants.GLOVE_FILE, self.tokenizer.word_index)
        with open(neural_constants.GLOVE_FILE) as f:
            glove = load_glove(f, self.tokenizer.word_index)

        # compute some neural_constants
        vocab_size = len(self.tokenizer.word_index) + 1

        # set model parameters
        self.model = Sequential()
        model_layers = [
            # must have these two layers firsts
            layers.Embedding(vocab_size,
                             neural_constants.GLOVE_DIMENSIONS,
                             weights=[glove],
                             input_length=neural_constants.SLICE_LENGTH,
                             trainable=False),
            # now we have some options
            layers.GlobalMaxPool1D(),
            layers.Dense(35, activation="relu"),

            # probably want a final sigmoid layer to get smooth value in range (0, 1)
            layers.Dense(1, activation="sigmoid")
        ]
        # add them in
        for layer in model_layers:
            self.model.add(layer)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        """
        print(np.shape(X_train))
        print(np.shape(y_train))
        print(np.shape(X_validation))
        print(np.shape(y_validation))
        """

        #X_train, y_train = shuffle_parallel_arrays(X_train, y_train)

        # now we fit (can take a while)
        self.model.fit(X_train, y_train,
                       epochs=5,
                       verbose=False,
                       shuffle=True,
                       validation_data=(X_validation, y_validation),
                       batch_size=5)

        if neural_constants.DIAGNOSTIC_PRINTING:
            def cm(true, pred):
                m = confusion_matrix(true, pred)
                print("Confusion matrix")
                print("   {0:3s} {1:3s}".format("P+", "P-"))
                print("T+ {0:<3d} {1:<3d}".format(m[1][1], m[0][1]))
                print("T- {0:<3d} {1:<3d}".format(m[1][0], m[0][0]))

            y_train_pred = [round(x[0]) for x in list(self.model.predict(X_train, verbose=False))]
            y_validation_pred = [round(x[0]) for x in list(self.model.predict(X_validation, verbose=False))]

            loss, acc = self.model.evaluate(X_train, y_train, verbose=False)
            print("Train L/A: {0:.4f} {1:.4f}".format(loss, acc))
            cm(y_train, y_train_pred)

            print()

            loss, acc = self.model.evaluate(X_validation, y_validation, verbose=False)
            print("Validation L/A: {0:.4f} {1:.4f}".format(loss, acc))
            cm(y_validation, y_validation_pred)

        # now we fit (can take a while)
        self.model.fit(X_train, y_train,
                       epochs=10,
                       verbose=False,
                       shuffle=True,
                       validation_data=(X_validation, y_validation),
                       batch_size=10)

        if neural_constants.DIAGNOSTIC_PRINTING:
            loss, acc = self.model.evaluate(X_train, y_train, verbose=False)
            print("Train L/A: {0:.4f} {1:.4f}".format(loss, acc))

            loss, acc = self.model.evaluate(X_train, y_train, verbose=False)
            print("Train L/A: {0:.4f} {1:.4f}".format(loss, acc))


    def predict(self, tokenized_file : str, minimum_confidence=.8):
        """

		:param tokenized_file: the array containing the ordered, sanitized word tokens from a single file
		:param minimum_confidence: the minimum confidence level required to the classifier to label a data point as
		any given class. Only used by applicable classifiers.
		:return: a list of tuples of [(class label, confidence)] for each class label where confidence >
		minimum_confidence. Confidence will be 1 for classifiers where confidence is not a normally used feature.
		"""

        raise NotImplementedError
