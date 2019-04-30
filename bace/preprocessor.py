from glob import glob, iglob
from nltk.corpus import stopwords
from typing import List, Iterable, Text, Container, Tuple, Optional, Dict
import numpy as np
import os
import pandas as pd
from argparse import ArgumentTypeError

# Type aliases for filter_texts function
Token = Text
Tokens_str = Token
Tokens = List[Token]


def filter_tokens(tokens: Iterable[Text],
                  stop_words: Optional[Container[Text]]) -> Tokens_str:
    """
    Filter out unnecessary tokens from given list of tokens.

    Parameters
    ----------
    tokens: Iterable[Text]
        The list of tokens to filter
    stop_words: Optional[Container[Text]] :
        If given, the set of stopwords to filter out form the tokens list.
    Returns
    -------
    Tokens_str
        The string version of all the tokens seperated by a single space.

    """
    from re import compile as regex
    from string import printable as printable_chars

    email_filter = regex(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    punc_filter = regex(r"[!#$&%()*+./:;<=>?@^_`{|}~,\-\\\"\']+")
    num_filter = regex("[0-9]+")
    enum_filter = regex(r"^\(?\w+([,.]\)?|[,.\)])")

    def yield_filtered_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
        """
        Yield all the filtered versions of the given tokens

        Parameters
        ----------
        tokens: Iterable[Token]
            The list of tokens to filter
        Yields
        ------
        Token
            The next filtered token in the tokens list

        """
        def filter_token(token: Text) -> Text:
            """
            Filter the given token

            Parameters
            ----------
            token: Text
                The token we want to filter
            Returns
            -------
            Token
                The filtered version of the current token
            """
            def strip_enum(token: Text) -> Text:
                """
                Remove any enumerations from the given token

                Parameters
                ----------
                token: Text :
                    The token that we want to remove any enumerations from
                Returns
                -------
                    A filtered version of the token that does not have any
                    enumerations.
                """
                if not token:
                    return ''
                if token[0] == '(' and token[len(token) - 1] != ')':
                    return ''
                if token[0] != '(' or (token[0] == '(' and token[len(token) -
                                                                 1] == ')'):
                    return ''.join(enum_filter.split(token))
                return ''

            if email_filter.match(token) or (
                stop_words and token in stop_words
            ):
                return ''
            # Strip enumeration from token
            token = strip_enum(token)
            # Strip punctuation from token
            token = ''.join(punc_filter.split(token))
            # Strip numbers from token
            token = ''.join(num_filter.split(token))
            # Remove non-printable characters
            token = ''.join(c for c in token if c in printable_chars)

            return '' if len(token) < 3 else token

        for token in tokens:
            filtered_token = filter_token(token)
            if filtered_token:
                yield filtered_token

    return ' '.join(token for token in yield_filtered_tokens(tokens))


def get_filtered_file(filename: Text,
                      stop_words: Optional[Container[Text]] = None
                      ) -> Tokens_str:
    """
    Get the filtered version of a single file.

    Parameters
    ----------
    filename: Text
        The name of the file we are reading from

    stop_words: Optional[Container[Text]] :
        Path of file consisting of extra stopwords to consider
        (Default value = None)

    Returns
    -------
    Tokens_str
        The string version of all the valid tokens in the file.

    """
    from re import compile as regex

    ws_filter = regex(r"\s+")
    with open(filename, 'rb') as f:
        decoded_str = f.read().decode(errors="ignore").strip().lower()
        return filter_tokens(ws_filter.split(decoded_str), stop_words)

    raise ValueError("Invalid File name!")


# def yield_filtered_files(should_export_extras: bool = False,
#                          input_dir: str = "data_clean",
def yield_filtered_files(input_dir: str = "data_clean",
                         stop_words: Optional[Container[Text]] = None
                         ) -> Iterable[pd.DataFrame]:
    """
    Yield the filtered version of all files corresponding to a specific label.

    Parameters
    ----------
    input_dir: str :
         The directory of where to find the folders that store the different
         folders filled with input data(Default value = "data_clean")
    stop_words: Optional[Container[Text]] :
        Path of file consisting of extra stopwords to consider
        (Default value = None)

    Yields
    ------
    pd.DataFrame
        A pandas DataFrame for the current label we have finished filtering.

    """

    # Note: output_dir should be the directory of where to store the individual
    # groupings of a single label's DataFrame file.

    # filtered_folder = os.path.abspath(output_dir)
    # if should_export_extras and not os.path.exists(filtered_folder):
    #     os.makedirs(filtered_folder)

    for folder_path in iglob(os.path.join(os.path.abspath(input_dir), "*")):
        valid_file_data: Dict[Text, Tokens_str] = {
            k: v for k, v in {
                file_name: get_filtered_file(file_name, stop_words)
                for file_name in glob(os.path.join(folder_path, "*.txt"))
            }.items()
            if v
        }

        if valid_file_data:
            folder_name = os.path.basename(folder_path)

            texts_df = pd.DataFrame({
                "filename": list(
                    os.path.basename(name) for name in valid_file_data.keys()
                ),
                "label": folder_name,
                "tokens": list(valid_file_data.values())
            })

            # if should_export_extras:
            #     export_folder = os.path.join(filtered_folder, folder_name)

            #     if not os.path.exists(export_folder):
            #         os.makedirs(export_folder)

            #     texts_df.to_csv(
            #         os.path.join(export_folder, "texts.csv"), index=False,
            #         columns=["filename", "tokens"]
            #     )

            yield texts_df


def split_dataset(export: str = "all",
                  split_percent: float = 0.8,
                  input_dir: str = "data",
                  output_dir: str = "filtered_data_clean",
                  stopwords_file_path: Optional[str] = None
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter and split all of the input data into a train split and a test split.

    Parameters
    ----------
    export: str :
        Flag to indicate which export mode to do. If "single" is chosen, only
        export a single csv file that holds the filtered texts of all files
        found in the input_dir. If "split" is chosen, only export the train set
        and test set into seperate csv files. If "all" is chosen, export all of
        the above.
        (Default value = "all")
    split_percent: float :
        The percent of the input data set that will go to the training set.
         (Default value = 0.8)
    input_dir: str :
        The directory that holds all of the data. Folder structure should be in
        the following form:
            data/
                label1/
                    data1.txt
                    data2.txt
                label2/
                    data3.txt
                    data4.txt
         (Default value = "data")
    output_dir: str :
        The directory that will hold the exported files.
         (Default value = "filtered_data_clean")
    stopwords_file_path: Optional[str] :
        Path of file consisting of extra stopwords to consider
        (Default value = None)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple where the first item is a pandas DataFrame of the training set
        and the second item is a pandas DataFrame of the test set.
    """

    if export == "single" or export == "all":
        full_arr: List[Tuple[str, str, str]] = []

    train_arr: List[Tuple[str, str, str]] = []
    test_arr: List[Tuple[str, str, str]] = []

    stop_words = set(stopwords.words('english'))
    if stopwords_file_path:
        with open(stopwords_file_path, "r") as fsmb_stop_words:
            stop_words.update(fsmb_stop_words.read().splitlines())

    for df in yield_filtered_files(input_dir=input_dir,
                                   stop_words=stop_words):

        if export == "single" or export == "all":
            full_arr.extend(df.values)

        sample_train = df.sample(frac=split_percent)
        train_arr.extend(sample_train.values)
        test_arr.extend(df.drop(sample_train.index).values)

    filtered_folder = os.path.abspath(output_dir)

    if not os.path.exists(filtered_folder):
        os.makedirs(filtered_folder)

    train_df = pd.DataFrame(train_arr,
                            columns=["filename", "label", "tokens"])
    test_df = pd.DataFrame(test_arr,
                           columns=["filename", "label", "tokens"])

    if export == "single" or export == "all":
        pd.DataFrame(full_arr, columns=["filename", "label", "tokens"]).to_csv(
            os.path.join(filtered_folder, "all_texts.csv"),
            index=False
        )
    if export == "split" or export == "all":
        train_df.to_csv(os.path.join(filtered_folder, "train_texts.csv"),
                        index=False
                        )
        test_df.to_csv(os.path.join(filtered_folder, "test_texts.csv"),
                       index=False
                       )

    return train_df, test_df


def get_slices(all_texts_df: pd.DataFrame,
               slice_length: int = 25,
               overlap_percent: float = 0) -> pd.DataFrame:
    """
    Split the given pandas DataFrame into many slices.

    Parameters
    ----------
    all_texts_df: pd.DataFrame :
        The pandas DataFrame that we are slicing. The formatting should match
        the structure that appears in the `split_dataset` function.
    slice_length: int :
        The maximum number of tokens any slice should be generated from the
        input DataFrame. (Default value = 25)
    overlap_percent: float :
        The number of tokens any adjacent slices can share. (Default value = 0)

    Returns
    -------
    pd.DataFrame
        pandas DataFrame that only holds the sliced version of the data and
        their corresponding label.
    """

    if overlap_percent >= 1 or overlap_percent < 0:
        raise ValueError("Invalid overlap amount")

    step = max(1, int(slice_length * (1 - overlap_percent)))

    all_slices: List[Tuple[Text, Text]] = []

    for row in all_texts_df.itertuples(index=False):
        tokens = row.tokens.split()
        snippets = [
            ' '.join(tokens[i:min(len(tokens), i + slice_length)])
            for i in range(0, len(tokens), step)
        ]
        all_slices += [(row.label, snippet) for snippet in snippets]

    return pd.DataFrame(all_slices, columns=["label", "slice"])


def export_fasttext_data(df: pd.DataFrame, output_name: str,
                         slice_length: Optional[int] = 25,
                         overlap_percent: float = 0) -> None:
    """
    Generate fasttext-compatible datasets

    Parameters
    ----------
    df: pd.DataFrame :
        The pandas DataFrame that will be used to derive the create the
        fasttext dataset.
    output_name: str :
        The name of the file that will be genreated
    slice_length: int :
        The maximum number of tokens any slice should be generated from the
        input DataFrame. (Default value = 25)
    overlap_percent: float :
        The number of tokens any adjacent slices can share. (Default value = 0)
    """

    if not os.path.exists(os.path.dirname(output_name)):
        os.makedirs(os.path.dirname(output_name))

    df["label"] = "__label__" + df["label"]
    df.drop(columns=["filename"], inplace=True)
    if slice_length:
        np.savetxt(
            output_name,
            get_slices(df, slice_length, overlap_percent).values,
            fmt="%s"
        )


def construct_parser_preprocessor(subparser) -> None:
    """
    Construct the preprocessor subparser.

    Parameters
    ----------
    subparser
        The subparser to modify that will include all necessary arguments to
        perform preprocessing.
    """

    def within_percent_interval(interval_str: str) -> float:
        """
        Checks whether or not the given string representation of a floating
        point number is in the interval [0, 1].

        Parameters
        ----------
        interval_str: str :
            The string that needs to be checked for a valid value

        Returns
        -------
        float
            If valid, the number representation of interval_str

        Raises
        ------
        ValueError
            If interval_str cannot be converted to a floating pointer number
        ArgumentTypeError
            If the number is not within the interval [0, 1]
        """
        interval = float(interval_str)
        if interval < 0 or interval > 1:
            raise ArgumentTypeError("Input given is out of bounds!")

        return interval
    """
    if subparser:
        preprocess_parser = subparser.add_parser(
            "preprocess",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help="Preprocess given dataset",
        )
    else:
        preprocess_parser = argparse.ArgumentParser(
            description='Preprocess given dataset',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    """

    subparser.add_argument(
        'input_dir', type=str, default="data_clean", metavar="input-dir",
        help='Input directory to preprocess'
    )

    subparser.add_argument(
        '-o', '--output-dir', type=str, default="filtered_data_clean",
        help='Output directory to hold preprocessed data_clean'
    )

    subparser.add_argument(
        '--stopwords', type=str, default=None,
        help='Path to the .csv stop words file'
    )

    # Make file generation options mutually exclusive
    # Note, all 3 of the flags appear. However, we only want 1 of them to
    # appear.
    subparser.add_argument(
        '--export', type=str, default="single",
        choices=["single", "split", "fasttext", "all"],
        help='Indicate whether you only want a single file holding all of the '
        'preprocessed data_clean, or both. If "split\" was chosen, it '
        'utilizes the "--train-split" argument to know how big to make the '
        'training and testing sets. If fasttext is given, it utilizes the '
        '"--slice-length" argument to know how big to make each slice'
    )

    subparser.add_argument(
        '--slice_length', type=int, default=25,
        help="Number of tokens to have per slice of a file."
    )

    subparser.add_argument(
        '--train-split', type=within_percent_interval, default=.8,
        metavar="[0-1]",
        help="Percentage in interval [0,1] of total data_clean going to the \
        training dataset."
    )

    subparser.set_defaults(run=run_preprocessor)


def run_preprocessor(args) -> None:
    """
    Run the preprocessor

    Parameters
    ----------
    args
        The NameSpace of command line arguments generated by the preprocessor
        argument parser
    """
    train_df, test_df = split_dataset(
        export=args.export,
        split_percent=args.train_split,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stopwords_file_path=args.stopwords
    )

    if args.export == "fasttext" or args.export == "all":
        train_slice_name = os.path.join(
            os.getcwd(), args.output_dir, "fasttext_train.txt"
        )
        test_slice_name = os.path.join(
            os.getcwd(), args.output_dir, "fasttext_test.txt"
        )
        export_fasttext_data(
            train_df, train_slice_name, slice_length=args.slice_length
        )
        export_fasttext_data(
            test_df, test_slice_name, slice_length=args.slice_length
        )