import pandas as pd
from sys import argv
from typing import List, Text, Tuple
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from os.path import join as path_join

def read_data(filename: Text) -> Tuple[List[Text], List[Text]]:
    dataset = pd.read_csv(filename)
    return dataset['label'].tolist(), dataset['tokens'].tolist()

def bag_of_words(train_labels: List[Text], train_data:List[Text], test_labels: List[Text], test_data:List[Text], num: int = 200):
    classifier = GaussianNB()
    classifier.fit(CountVectorizer(max_features = num).fit_transform(train_data).toarray(),train_labels)
    class_prediction = classifier.predict(CountVectorizer(max_features = num).fit_transform(test_data).toarray())

    print(confusion_matrix(test_labels,class_prediction))
    print(classification_report(test_labels,class_prediction))
    print(accuracy_score(test_labels,class_prediction))

if __name__ == "__main__":
    num_feats = int(argv[1]) if len(argv) > 1 else 200
    train_file_name = path_join("filtered_data","train_texts.csv")
    test_file_name = path_join("filtered_data","test_texts.csv")

    labels_train, data_train = read_data(train_file_name)
    labels_test, data_test = read_data(test_file_name)
    bag_of_words(labels_train,data_train,labels_test,data_test,num_feats)