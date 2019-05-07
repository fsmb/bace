**FSMB BACE Classifier**

This repository is for a board action classifier for the federation of state medical boards. This is a senior design
project by UTD computer science undergraduate students that will use natural-language processing and machine learning
techniques to classify documents based on limited training data.

The training and testing data will be in the form of text files that are cleaned as much as possible after being
extracted from the PDF documents that the various medical boards provides via OCR techniques.

The goal of this project is to, given some input training data, use various techniques to classify new data.

This project has a dependency on Fasttext, which may require separate installation.

**Notable Files**

* bace_driver.py : The main driver for the various tasks, including the preprocessor and the classifiers.
    Run
    > python bace_driver.py -h

    or

    > python3 bace_driver.py -h

    as appropriate to access help documentation.

* bace/ : Modules that bace_driver.py depends upon
    * sample_commands.txt : A selection of example commands that assume that you have the files mentioned blow in root:
    * classifiers/ : the modules for the classification tasks
        * bayesian/ : the module for a context-insensitive, bag-of-words based, bayesian classifier.
        * fasttext/ : the module for a context-sensitive classifier based on the fasttext library
        * neural/ : the module for a context-sensitive neural solution that uses GLoVE word vectory
        to represent component words
    * preprocessor.py : the module that handles document preprocessing and performs limited document cleaning
* bace_data/data/ : A collection of directories containing any text documents. Assumed to exist by sample_commands.
* bace_data/data_clean/ : A collection of directories containing cleaned text documents. Assumed to exist by sample_commands.
* bace_data/nn/ : Much like data_clean, mentioned above, but potentially with files removed. Assumed to exist by sample_commands.
* bace_data/nn_test/ : A directory holding a collection of files taken from the subdirectories of nn. Assumed to exist by sample_commands.

**Glossary**

Accuracy : The percent examples from some set that a classifier predicts the correct label(s) for.

Bag of words : A method of representing a document as a feature vector of counts of words. Order and context is not
preserved.

Bayesian : A descriptor applied to models and techniques, typically those with statistical techniques.

Classifier : A system that uses a model to predict the classes of data points. IE, given an input piece of data X, it
 will produce some subset of all possible class labels Y that is predicted as the class or classes of X.

Data : A collection of information used to train, test, and apply a model. In this application, data takes the form
of plain text medical board action report documents.

Imbalanced data : Data where the classes are not equally probable. Can lead to issues where models will overpredict
some subset of the classes, and still achieve high accuracy, as they only get rarer classes wrong.

Model : Some representation of the properties of the data. For learning applications, models typically take the form
either a function f(x) = y, or some probability distribution on y and x.

Neural Network : A type of model, loosely based on the brains of living things. A neural network typically consists
of a directed graph with weighted edges, some number of inputs, and some number of outputs. The edge weights are
learned and thereby represent some unknown function. Larger networks can represent more complicated functions, but
can also risk memorizing the training data, also known as overfitting.

Overfitting : When a model has effectively memorized the training data. An overfit model will not generalize well. A
sign of overfitting is substantially higher accuracy on the training data as opposed to the test data.

Preprocessor : A program applied to data before it is used to train or test a model. Typically, preprocessors will
format the data and perform some form of cleaning.

Word embeddings : A dense real number vector representing some word, typically with 300 or fewer dimensions. Word
vectors represent word meaning in a mathematically significant way. Many pre-trained sets of word embeddings exist.
Fasttext and GLoVe are both popular examples.

**Future Optimizations**

As more data comes in, there are a few key optimizations that will be called for:
 * Tuning the number of features for the bayesian model.
 * Tuning the layers of the neural network model (found in the train function). As data increases, more complicated
 neural models may be appropriate, in the form of more layers or more complicated layers.
 * Tuning the slice length to find the best performance for the various classifiers.

**Citations**

*FastText*

>A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

*GLoVe*

>Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
>https://nlp.stanford.edu/pubs/glove.pdf
