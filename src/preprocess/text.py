import re
import itertools
from turtle import st
import contractions
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.corpus import words
import pandas as pd
from pandarallel import pandarallel
import wordninja
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from time import time

from sklearn.model_selection import train_test_split

pandarallel.initialize()
words_list = words.words()
RANDOM_SEED = 100


class DataSet(object):
    def __init__(self, dataframe):
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(
            dataframe
        )

    def split_data(self, dataframe):
        return train_test_split(
            dataframe["comment_text_array_norm"].to_list(),
            dataframe["toxic"],
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=dataframe["toxic"],
        )


class TextDataProcessor(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def read_textdata(self):
        with open(self.filepath, "r") as f:
            return pd.read_csv(f)

    def run(
        self,
        norm_type="stemming",
        subtype="porter",
        simple_preprocess=True,
        for_bert=False,
        load_ready=True,
        save=False,
    ):
        if norm_type == "lemma":
            subtype = ""

        if load_ready:
            type_of_preprocess = "simple_" if simple_preprocess else ""
            bert_prefix = "for_bert_" if for_bert else ""
            return pd.read_csv(
                f"../data/{type_of_preprocess}preprocessed_{bert_prefix}{norm_type}_{subtype}.csv"
            )

        nltk.download("wordnet")
        nltk.download("omw-1.4")
        nltk.download("words")
        dataframe = self.drop_empty_arrays_data(
            self.make_normalization(
                self.clean_text_as_list(
                    self.clean_text_as_string(
                        self.read_textdata()[["comment_text", "toxic"]]
                    ),
                    simple=simple_preprocess,
                    for_bert=for_bert,
                ),
                norm_type=norm_type,
                subtype=subtype,
            )
        )
        if save:
            type_of_preprocess = "simple_" if simple_preprocess else ""
            bert_prefix = "for_bert_" if for_bert else ""
            name = f"../data/{type_of_preprocess}preprocessed_{bert_prefix}{norm_type}_{subtype}.csv"
            dataframe.to_csv(name)
        return dataframe

    @staticmethod
    def clean_text_as_string(dataframe):
        start = time()
        regex_pat = re.compile(r"[^\w\s\']", flags=re.IGNORECASE)
        dataframe["comment_text_cleaned"] = (
            (
                dataframe["comment_text"]
                .str.replace(
                    r'https?://[^\s<>"]+|www\.[^\s<>"]+', "", regex=True
                )  # remove web addresses
                .str.replace(r"(\s[\'\"])|([\'\"]\s)", " ", regex=True)  # remove quotes
                .str.replace(
                    regex_pat, "", regex=True
                )  # leave only letters, spaces and apostrophes
                .str.replace(r"\d", "", regex=True)
                .str.replace("\n", " ", regex=False)  # remove next line characters
                .str.replace(r"\s+", " ", regex=True)  # remove multiple spaces
                .str.replace("_", " ", regex=False)
            )
            .str.lower()  # lower case
            .str.strip()  # remove spaces from borders
        )

        print("String cleaning techniques done :::", time() - start)
        return dataframe

    @staticmethod
    def clean_text_as_list(dataframe, simple=True, for_bert=False):
        start = time()

        if simple:
            dataframe["comment_text_array"] = (
                dataframe["comment_text_cleaned"]
                .str.split(" ")
                .parallel_apply(drop_unknown_and_stop_word)
            )
        else:

            if for_bert:
                dataframe["comment_text_array"] = (
                    dataframe["comment_text_cleaned"]
                    .str.split(" ")
                    .parallel_apply(transform_text_list_for_bert)
                )
            else:
                dataframe["comment_text_array"] = (
                    dataframe["comment_text_cleaned"]
                    .str.split(" ")
                    .parallel_apply(transform_text_list)
                )
        print("List cleaning techniques done :::", time() - start)
        return dataframe

    @staticmethod
    def make_normalization(dataframe, norm_type, subtype):
        start = time()
        if norm_type == "stemming":
            if subtype == "porter":
                stemmer = PorterStemmer()
            elif subtype == "lancaster":
                stemmer = LancasterStemmer()
            else:
                raise ValueError("Unknown stemmer type: %s" % subtype)
            dataframe["norm_type"] = norm_type + "_" + subtype
            dataframe["comment_text_array_norm"] = dataframe[
                "comment_text_array"
            ].apply(lambda lst: " ".join([stemmer.stem(x) for x in lst]))

        elif norm_type == "lemma":
            lemmatizer = WordNetLemmatizer()
            dataframe["norm_type"] = norm_type
            dataframe["comment_text_array_norm"] = dataframe[
                "comment_text_array"
            ].apply(lambda lst: " ".join([lemmatizer.lemmatize(x) for x in lst]))
        else:
            raise ValueError("Unknown normalizing type: %s" % norm_type)
        print("Normalisation done :::", time() - start)
        return dataframe

    @staticmethod
    def drop_empty_arrays_data(dataframe):
        return dataframe[
            dataframe["comment_text_array_norm"].apply(lambda lst: len(lst) > 0)
        ]


def transform_text_list(text_list):
    text_list = remove_non_latin_characters(text_list)
    text_list = fix_contractions(text_list)
    text_list = remove_mult_equal_letters(text_list)
    text_list = remove_repeating_patterns(text_list)
    text_list = remove_stopwords_from_text_list(text_list)
    text_list = remove_too_short_words(text_list)
    text_list = remove_too_long_words(text_list)

    return text_list


def transform_text_list_for_bert(text_list):
    text_list = remove_non_latin_characters(text_list)
    text_list = fix_contractions(text_list)
    text_list = remove_mult_equal_letters(text_list)
    text_list = remove_repeating_patterns(text_list)

    return text_list


def fix_contractions(text_list):
    #  fix contractions
    return [
        contractions.fix(word)
        for word in text_list
        if not "'" in contractions.fix(word)
    ]


def remove_mult_equal_letters(text_list):
    # remove multiple equal letters
    return [
        word if word in words_list else "".join([c[0] for c in itertools.groupby(word)])
        for word in text_list
    ]


def remove_repeating_patterns(text_list):
    # remove reapeating patterns
    return [re.sub(r"\b(.+?)\1+|(.+?)\1+", r"\1", word) for word in text_list]


def remove_non_latin_characters(text_list):
    # remove non-latin characters
    return [re.sub(r"[^\x00-\x7f]", "", word) for word in text_list]


def remove_stopwords_from_text_list(text_list):
    return remove_stopwords(" ".join(text_list)).split(" ")


def remove_too_short_words(text_list, short_thld=1):
    return [word for word in text_list if len(word) > short_thld]


def remove_too_long_words(text_list, long_thld=20):
    return [word for word in text_list if len(word) < long_thld]


def drop_unknown_and_stop_word(lst, voc=words_list):
    return [
        remove_stopwords(word)
        for word in lst
        if (word in voc and len(remove_stopwords(word)) > 2)
    ]
