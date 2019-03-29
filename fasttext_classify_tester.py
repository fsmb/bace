from os.path import join as path_join
import fasttext as ft


if __name__ == "__main__":
    train_file_name = path_join("filtered_data", "fasttext_train.txt")
    test_file_name = path_join("filtered_data", "fasttext_test.txt")

    classifier = ft.supervised(train_file_name, "model")

    results = classifier.test(test_file_name)
    print(results.precision)
    print(results.recall)

    prediction = classifier.predict(["convict"], k=3)
    print(prediction)
