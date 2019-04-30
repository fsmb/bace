import bace.classifiers.neural.neural_constants as neural_constants



def run_nn(args):
    """A function used to run the neural classifier based on the args

    Parameters
    ----------
    args : Namespace
        A namespace from an argparser using construct_parser_nn

    """
    # import here bc some imports are heavy
    import bace.classifiers.neural.neural_classifier as nc
    from os import listdir
    import pickle
    from argparse import ArgumentError
    from random import shuffle

    def child_path(dir, fname):
        return dir + '/' + fname

    if args.load_saved is not None:
        with open(args.load_saved, 'r') as f:
            clf = pickle.load(f)
    elif args.input:
        classes = listdir(args.input)

        ids = [None] * len(classes)
        for i in range(len(classes)):
            directory = child_path(args.input, classes[i])
            files = list(filter(lambda x : x.endswith(".txt"), listdir(directory)))
            ids[i] = (classes[i], directory, files)

        clf = nc.NeuralClassifier()
        for i in range(len(ids)):
            name, dir, init_inputs = ids[i]
            inputs = init_inputs.copy()
            shuffle(inputs)

            if args.evaluate:
                if not (0 < args.evaluate < 1):
                    raise Exception("Evaluation percent must be in range (0, 1)")
                num_training = int(args.evaluate * len(inputs))
            else:
                num_training = len(inputs)
            X_train = inputs[:num_training]

            if args.evaluate:
                X_validate = inputs[num_training+1:]

                for fname in X_train:
                    with open(child_path(dir, fname), encoding="windows-1252") as f:
                        clf.add_data("{0} {1}".format(name, fname), f.read(), name)
                for fname in X_validate:
                    with open(child_path(dir, fname), encoding="windows-1252") as f:
                        clf.add_validation_data("{0} {1}".format(name, fname), f.read(), name)
            else:
                for fname in X_train:
                    with open(child_path(dir, fname), encoding="windows-1252") as f:
                        clf.add_data("{0} {1}".format(name, fname), f.read(), name)

        if args.evaluate:
            clf.train(max_number_tokens=args.num_tokens,
                      glove_file=args.glove_embedding,
                      glove_dimensions=args.glove_dimensions,
                      num_epochs=args.epochs,
                      diagnostic_printing=True,
                      batch_size=args.batch_size
                      )
        elif args.pickle:
            clf.train(max_number_tokens=args.num_tokens,
                      glove_file=args.glove_embedding,
                      glove_dimensions=args.glove_dimensions,
                      num_epochs=args.epochs,
                      diagnostic_printing=False,
                      batch_size=args.batch_size,
                      slice_length=args.slice_length,
                      slice_overlap=args.slice_overlap
                      )

            clf.pickle(args.pickle, keep_data=False)
        elif args.slice:
            clf.train(max_number_tokens=args.num_tokens,
                      glove_file=args.glove_embedding,
                      glove_dimensions=args.glove_dimensions,
                      num_epochs=args.epochs,
                      diagnostic_printing=False,
                      batch_size=args.batch_size,
                      slice_length=args.slice_length,
                      slice_overlap=args.slice_overlap
                      )


            if not (0 <= args.slice_overlap < .5):
                raise ArgumentError('Slice overlap must be in range [0, .5)')
            with open(args.slice) as f:
                results = clf.slice_and_predict(f.read(),
                                                slice_length=args.slice_length,
                                                slice_overlap=args.slice_overlap)
            for label, str in results:
                print(label, ':')
                print(str)

        elif args.test_folder:
            clf.train(max_number_tokens=args.num_tokens,
                      glove_file=args.glove_embedding,
                      glove_dimensions=args.glove_dimensions,
                      num_epochs=args.epochs,
                      diagnostic_printing=False,
                      batch_size=args.batch_size,
                      slice_length=args.slice_length,
                      slice_overlap=args.slice_overlap
                      )
            for fname in listdir(args.test_folder):
                with open(child_path(args.test_folder, fname)) as f:
                    p = clf.predict(fname,
                                    slice_length=args.slice_length,
                                    slice_overlap=args.slice_overlap)
                    print(fname, p)
            X_validate = inputs[num_training+1:]

            for fname in X_train:
                with open(child_path(dir, fname), encoding="windows-1252") as f:
                    clf.add_data("{0} {1}".format(name, fname), f.read(), name)
            for fname in X_validate:
                with open(child_path(dir, fname), encoding="windows-1252") as f:
                    clf.add_validation_data("{0} {1}".format(name, fname), f.read(), name)

        clf.train(max_number_tokens=args.num_tokens,
                      glove_file=args.glove_embedding,
                      glove_dimensions=args.glove_dimensions,
                      num_epochs=args.epochs,
                      diagnostic_printing=False,
                      batch_size=args.batch_size,
                      slice_length=args.slice_length,
                      slice_overlap=args.slice_overlap
                  )
    else:
        raise Exception("Missing mandatory input arg - this error should be impossible")


def construct_parser_nn(subparser):
    """Adds the arguments necessary for run_nn to the give parser, and adds a functional reference to
        run_nn as the run argument as the value of the run attribute of the parser

    Parameters
    ----------
    subparser
        An argparse object that supports arguments, mutually exclusion, and defaults

    """
    subparser.add_argument(
        'glove_embedding', type=str,
        metavar='glove_file',
        help='Path to glove embedding file and its number of dimensions'
    )

    subparser.add_argument(
        'glove_dimensions', type=int,
        metavar='glove_dimensions',
        help='Number of dimensions of glove file'
    )

    input = subparser.add_mutually_exclusive_group(required=True)

    input.add_argument(
        '-i', '--input', type=str, metavar="input-dir",
        help='Path to input training data'
    )
    input.add_argument(
        '-l', '--load_saved', type=str,
        help='Path to saved classifier produced with -s option',
    )

    subparser.add_argument(
        '-o', '--output_dir', type=str, default="results",
        help='Output directory to hold bow classifier output files'
    )

    output = subparser.add_mutually_exclusive_group(required=True)

    output.add_argument(
        '-t', '--test_folder', type=str,
        help='Directory containing directories containing text files'
    )
    output.add_argument(
        '-e', '--evaluate', type=float, metavar="(0-1)",
        help='split and evaluate on input folder with the given percent'
    )
    output.add_argument(
        '-p', '--pickle', type=str,
        help='saves the model to a given file location with pickle'
    )
    output.add_argument(
        '-s', '--slice', type=str,
        metavar="file",
        help='slices the given file into slices of a given length with the given overlap'
    )

    subparser.add_argument(
        '--slice_length', type=int, default=neural_constants.SLICE_LENGTH,
        help='number of tokens in the slices'
    )
    subparser.add_argument(
        '--slice_overlap', type=float, default=neural_constants.SLICE_OVERLAP,
        metavar='[0-1)',
        help='percent of the slice that is overlapping with adjacent slices (half on each side)'
    )
    subparser.add_argument(
        '-n', '--num_tokens', type=int, default=neural_constants.MAX_NUMBER_TOKENS,
        help='maximum number of tokens, will increase as data increases and number of classes increases'
    )
    subparser.add_argument(
        '--epochs', type=int, default=neural_constants.NUM_EPOCHS,
        help='maximum number of tokens, will increase as data increases and number of classes increases'
    )
    subparser.add_argument(
        '-b', '--batch_size', type=int, default=neural_constants.MAX_NUMBER_TOKENS,
        help='maximum number of tokens, will increase as data increases and number of classes increases'
    )
    subparser.set_defaults(run=run_nn)