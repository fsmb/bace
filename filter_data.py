from glob import glob, iglob
from nltk.corpus import stopwords
from typing import List, Iterable, Text, Container, Tuple
import numpy as np
import os
import pandas as pd

# Type aliases for filter_texts function
Token = Text
Tokens_str = Token
Tokens = List[Token]


def filter_texts(filenames: Iterable[Text],
                 stop_words: Container[Text]) -> List[Tokens_str]:
    from re import compile as regex
    from string import printable as printable_chars

    ws_filter = regex(r"\s+")
    email_filter = regex(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    punc_filter = regex(r"[!#$&%()*+./:;<=>?@^_`{|}~,\-\\\"\']+")
    num_filter = regex("[0-9]+")
    enum_filter = regex(r"^\(?\w+([,.]\)?|[,.\)])")

    def files(filenames: Iterable[Text]) -> Iterable[Tokens]:
        for filename in filenames:
            with open(filename, 'rb') as f:
                decoded_str = f.read().decode(errors="ignore").strip().lower()
                yield ws_filter.split(decoded_str)

    def yield_filtered_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
        def filter_token(token: Text) -> Text:
            def strip_enum(token: Text) -> Text:
                if not token:
                    return ''
                if token[0] == '(' and token[len(token) - 1] != ')':
                    return ''
                if token[0] != '(' or (token[0] == '(' and token[len(token) -
                                                                 1] == ')'):
                    return ''.join(enum_filter.split(token))
                return ''

            if token in stop_words or email_filter.match(token):
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

    return list(filter(None, [
        ' '.join(token for token in yield_filtered_tokens(tokens))
        for tokens in files(filenames)
    ]))


def yield_files(should_export: bool = False) -> Iterable[pd.DataFrame]:
    def squash_texts(texts: List[Tokens_str]) -> List[Token]:
        return [
            token
            for tokens in texts
            for token in tokens.split()
        ]

    stop_words = set(stopwords.words('english'))
    with open("Distinct Combined Stop Words.csv", "r") as fsmb_stop_words:
        stop_words.update(fsmb_stop_words.read().splitlines())

    filtered_folder = os.path.join(os.getcwd(), "filtered_data")
    if should_export and not os.path.exists(filtered_folder):
        os.makedirs(filtered_folder)

    for folder_path in iglob(os.path.join(os.getcwd(), "data", "*")):
        full_file_names = glob(os.path.join(folder_path, "*.txt"))
        texts = filter_texts(full_file_names, stop_words)

        if texts:
            folder_name = os.path.basename(folder_path)
            export_folder = os.path.join(filtered_folder, folder_name)
            if should_export and not os.path.exists(export_folder):
                os.makedirs(export_folder)

            base_names = [os.path.basename(name) for name in full_file_names]

            texts_df = pd.DataFrame({
                "filename": base_names,
                "label": folder_name,
                "tokens": texts
            })

            if should_export:
                texts_df.to_csv(os.path.join(export_folder, "texts.csv"),
                                index=False)

            yield texts_df


def split_dataset(export_extras: bool = False, split_percent: float = 0.8
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if export_extras:
        full_arr: List[Tuple[str, str, str]] = []

    train_arr: List[Tuple[str, str, str]] = []
    test_arr: List[Tuple[str, str, str]] = []

    for df in yield_files(export_extras):
        if export_extras:
            full_arr.extend(df.values)

        sample_train = df.sample(frac=split_percent)

        train_arr.extend(sample_train.values)
        test_arr.extend(df.drop(sample_train.index).values)

    train_df = pd.DataFrame(train_arr,
                            columns=["filename", "label", "tokens"])
    test_df = pd.DataFrame(test_arr,
                           columns=["filename", "label", "tokens"])

    filtered_folder = os.path.join(os.getcwd(), "filtered_data")
    if export_extras:
        if not os.path.exists(filtered_folder):
            os.makedirs(filtered_folder)

        pd.DataFrame(full_arr, columns=["filename", "label", "tokens"]).to_csv(
            os.path.join(filtered_folder, "all_texts.csv"),
            index=False
        )
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
                         slice_length: int = 25, overlap_percent: float = 0):

    if not os.path.exists(os.path.dirname(output_name)):
        os.makedirs(os.path.dirname(output_name))

    df["label"] = "__label__" + df["label"]
    df.drop(columns=["filename"], inplace=True)
    np.savetxt(output_name,
               get_slices(df, slice_length, overlap_percent).values,
               fmt="%s")


if __name__ == "__main__":
    train_df, test_df = split_dataset(export_extras=True, split_percent=0.8)

    train_slice_name = os.path.join(os.getcwd(), "filtered_data",
                                    "fasttext_train.txt")
    test_slice_name = os.path.join(os.getcwd(), "filtered_data",
                                   "fasttext_test.txt")

    export_fasttext_data(train_df, train_slice_name, slice_length=10)
    export_fasttext_data(test_df, test_slice_name, slice_length=10)
