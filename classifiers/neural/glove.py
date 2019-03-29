import classifiers.neural.neural_constants as constants
import numpy as np

from typing import Dict

def load_glove(fname : str, word_index : Dict[str, int]):
    """
	Loads the dictionary of glove_embeddings vectors from a given input file

	:param f: a Glove file, open for reading
	:return: a dict of {word : vector, ...} for each word line in the file, where vector is a float array
	"""

    with open(fname , encoding="utf8") as f:
        data = np.zeros((len(word_index) + 1, constants.GLOVE_DIMENSIONS))
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                data[word_index[word]] = np.array(vector, dtype=np.float32)
        return data