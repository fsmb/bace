"""
Constants for use by the neural classifier
"""

# data_clean slicing
SLICE_LENGTH = 300
SLICE_OVERLAP = .2

# glove embeddings neural_constants
GLOVE_FILE = "classifiers/neural/glove_embeddings/glove.6B.100d.txt"
#GLOVE_FILE = "classifiers/neural/glove_embeddings/glove.6B.50d.txt"
GLOVE_DIMENSIONS = 100


# neural_constants for use by the neural classifier
MAX_NUMBER_TOKENS = 15000

# default # of epochs
NUM_EPOCHS = 10

# if neural model should do diagnostic printing
DIAGNOSTIC_PRINTING = True

# if it should impose balance
IMPOSE_BALANCE = False