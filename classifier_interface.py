from abc import ABC
import pandas as pd


class ClassifierWrapper(ABC):

	@abstractmethod
	def add_data(self, file_id, tokenized_file, true_label):
		"""

		:param file_id: a hashable ID for this particular file
		:param tokenized_file: a
		:param true_label:
		:return: None
		"""
		raise NotImplementedError

	@abstractmethod
	def get_data(self):
		"""

		:return: A structure [(file_id, tokenized_file, true_label),...] for all data added to this classifier with
		the add_data method
		"""
		raise NotImplementedError

	@abstractmethod
	def train(self):
		"""
		This classifier object will train on all the data that has been added to it using the adddata method
		:return:
		"""
		raise NotImplementedError

	@abstractmethod
	def predict(self, tokenized_file, minimum_confidence=.8):
		"""

		:param tokenized_file: the array containing the ordered, sanitized word tokens from a single file
		:param minimum_confidence: the minimum confidence level required to the classifier to label a data point as
		any given class. Only used by applicable classifiers.
		:return: a list of tuples of [(class label, confidence)] for each class label where confidence >
		minimum_confidence. Confidence will be 1 for classifiers where confidence is not a normally used feature.
		"""
		raise NotImplementedError