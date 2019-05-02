from bace.classifiers.neural.neural_classifier import NeuralClassifier as NC
from bace.classifiers.neural.comparative_neural_classifier import ComparativeNeuralClassifier as CNC
import bace.classifiers.neural.neural_constants as neural_constants

from os import listdir

from random import shuffle

def main():
    """Old tester, no longer functional

    """
    # no longer functional
    vals = dict()
    minpts = float('inf')
    for name, dir, id in data_constants.TYPES:
        l = listdir(dir)
        vals[id] = (name, dir, list(filter(lambda x : x.endswith(".txt"), l)))
        minpts = min(minpts, len(l))

    cnc = CNC()

    for id in vals:
        name, dir, init_inputs = vals[id]
        inputs = init_inputs.copy()
        shuffle(inputs)

        num_training = int(.8 * len(inputs))
        if neural_constants.IMPOSE_BALANCE:
            num_training = min(minpts, num_training)
        X_train = inputs[:num_training]
        X_test = inputs[num_training + 1:]

        for fname in X_train:
            with open(dir + fname, encoding="windows-1252") as f:
                cnc.add_data("{0} {1}".format(name, fname), f.read(), id)
        for fname in X_test:
            with open(dir + fname, encoding="windows-1252") as f:
                cnc.add_validation_data("{0} {1}".format(name, fname), f.read(), id)

    cnc.train()

    """
    for target_id in vals:
        print("CLASSIFIER ON {0}".format(vals[target_id][0]))
        nc = NC()

        for id in vals:
            name, dir, init_inputs = vals[id]
            inputs = init_inputs.copy()
            shuffle(inputs)
            num_training = int(.8 * len(inputs))
            if neural_constants.IMPOSE_BALANCE:
                num_training = min(minpts, num_training)
            X_train = inputs[:num_training]
            X_test = inputs[num_training+1:]

            for fname in X_train:
                with open(dir+fname, encoding="windows-1252") as f:
                    nc.add_data("{0} {1}".format(name, fname), f.read(), 1 if id == target_id else 0)
            for fname in X_test:
                with open(dir+fname, encoding="windows-1252") as f:
                    nc.add_validation_data("{0} {1}".format(name, fname), f.read(), 1 if id == target_id else 0)

        nc.train()
        print()
    """


if "__main__" == __name__:
    main()