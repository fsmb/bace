import pandas as pd
from typing import List, Text, Tuple
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
#from bace.classifiers.classifier import Classifier
from os.path import join as path_join
import argparse

def read_data(filename: Text) -> Tuple[List[Text], List[Text], List[Text]]:
    """
    reads the text file and extract the columns from the .csv file

    Parameters
    ----------
    filename: Text
        .csv file containing filenames, class, and tokens for each file

    Returns
    -------
    Tuple[List[Text], List[Text], List[Text]]
        returns the list of the filenames, class labels, and tokens in three separate lists
    """
    dataset = pd.read_csv(filename)
    return dataset['filename'].tolist(),\
           dataset['label'].tolist(), \
           dataset['tokens'].tolist()


def get_classifier(train_labels: List[Text], train_data: List[Text], num_features: int = 200):
    """
    trains the bag of words model based on the training labels and the training data, with a given number of features

    Parameters
    ----------
    train_labels: List[Text]
        List of the class labels for the training set
        
    train_data: List[Text]
        List of the tokens for the training set
        
    num_features: int
         number of most frequent tokens to store in model
         (Default value = 200)

    Returns
    -------
    classifier: GaussianNB
        model that fits Gaussian Naive Bayes according to Bag of Words
    """
    classifier = GaussianNB()
    classifier.fit(CountVectorizer(max_features = num_features).fit_transform(train_data).toarray(),train_labels)
    return classifier


def predict(classifier: GaussianNB, test_data:List[Text], num: int = 200):
    """
    predicts the class label probabilities for the test set based on the test tokens and given number of features

    Parameters
    ----------
    classifier: GaussianNB
        model that holds the bag of words model
        
    test_data: List[Text]
        list of the tokens for the test set
        
    num: int
        number of most frequent tokens to store in model
        (Default value = 200)

    Returns
    -------
    array-like
        Returns probabilities for each class in the model
    """
    return classifier.predict_proba(CountVectorizer(max_features=num).fit_transform(test_data).toarray())

def predict_single(classifier: GaussianNB, test_data:List[Text], num: int = 200):
    """
    predicts the most probable class value for the given test tokens and model

    Parameters
    ----------
    classifier: GaussianNB
        model that holds the bag of words model
        
    test_data: List[Text]
        list of the tokens for the test set
        
    num: int
        number of most frequent tokens to store in model
        (Default value = 200)

    Returns
    -------
    array
        Predicted target values for each file
    """
    return classifier.predict(CountVectorizer(max_features=num).fit_transform(test_data).toarray())

def show_metrics(test_labels: List[Text], class_prediction: List[Text]):
    """
    prints the statistics for the prediction results
    confusion_matrix:       n x n array
                            number value on the diagonal: (0,0), (1,1) ... (n,n): number of correct classifications
                            else: number of incorrect classifications
    classification_report:  report on several statistics
    accuracy_score:         decimal on how much is correct

    Parameters
    ----------
    test_labels: List[Text]
        List of the labeled class labels for the test set
        
    class_prediction: List[Text]
        List of the predicted class labels for the test set
    """
    print(confusion_matrix(test_labels,class_prediction))
    print(classification_report(test_labels,class_prediction))
    print("Accuracy:", accuracy_score(test_labels,class_prediction))

def run_bagofwords(args):
    """
    parser the subparser and runs the bag of words based on the arguments given

    Parameters
    ----------
    args
        variable length argument list
    """
    train_fnames, train_labels, train_tokens = read_data(args.training_file)
    test_fnames, test_labels, test_tokens = read_data(args.test_file)

    le = preprocessing.LabelEncoder().fit(train_labels)

    clf = get_classifier(train_labels, train_tokens, args.num_features)

    predictions = predict(clf, test_tokens, num=args.num_features)

    if args.slice:
        tokens = test_tokens[args.slice].split()
        slices = [tokens[x:x + 100] for x in range(0, len(tokens), 100)]
        slices = [' '.join(slices[i]) for i in range(len(slices))]
        predict_file = predict_single(clf, slices, num=args.num_features)
        print(test_fnames[args.slice])
        for i in range(len(predict_file)):
            print(predict_file[i], ":")
            print(slices[i])
    elif args.metrics:
        predictions = predict_single(clf, test_tokens, num=args.num_features)
        show_metrics(test_labels, predictions)
    else:
        for i in range(len(predictions)):
            print(test_fnames[i],end=' ')
            for j in range(len(predictions[i])):
                print(le.classes_[j] + ": " + str(predictions[i][j]),end = ' ')
            print()

def construct_parser_bow(subparser):
    """
    Construct Bag of Words subparser

    Parameters
    ----------
    subparser
        parser to hold the flags for the bag of words
    """

    """
        if subparser:
        bow_parser = subparser.add_parser(
            "bow",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help="Bag of words Classifier"
        )
    else:
        bow_parser = argparse.ArgumentParser(
            description='Bag of words Classifier',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    """
    #subparser.set_defaults(description="BAG OF WORDS ASDJFAHEKJD")

    subparser.add_argument(
        'training_file', type=str, default="data_clean", metavar="input-dir",
        help='Path to training .csv'
    )
    subparser.add_argument(
        '-o', '--output_dir', type=str, default="results",
        help='Output directory to hold bow classifier output files'
    )
    subparser.add_argument(
        '-t', '--test_file', type=str, default="test_texts.csv",
        help='Path to test .csv or .csv to predict'
    )
    subparser.add_argument(
        '-n', '--num_features', type=int, default=200,
        help='number of features to use in bag of words classification'
    )

    # Make results showing options mutually exclusive
    subparser.add_argument(
        '-m', '--metrics', action="store_true", default=False,
        help="Flag to just show metrics instead of predictions"
    )

    subparser.add_argument(
        '-s', '--slice', type=int, metavar="i",
        help="Flag to label slices of the ith document in the test .csv"
    )
    subparser.set_defaults(run=run_bagofwords)

"""
from bace.classifiers.classifier import Classifier

class BagOfWords(Classifier):
    def read_data(filename: Text) -> Tuple[List[Text], List[Text]]:
        dataset = pd.read_csv(filename)
        return dataset['label'].tolist(), dataset['tokens'].tolist()

    def get_classifier(train_labels: List[Text], train_data:List[Text], num: int = 500):
        classifier = GaussianNB()
        classifier.fit(CountVectorizer(max_features = num).fit_transform(train_data).toarray(),train_labels)
        return classifier

    def predict(classifier: GaussianNB, test_data:List[Text], num: int = 200):
        return classifier.predict(CountVectorizer(max_features = num).fit_transform(test_data).toarray())

    def show_metrics(test_labels: List[Text], class_prediction: List[Text]):
        print(confusion_matrix(test_labels,class_prediction))
        print(classification_report(test_labels,class_prediction))
        print(accuracy_score(test_labels,class_prediction))
"""