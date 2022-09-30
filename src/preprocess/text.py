import re
import itertools
import contractions
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
import pandas as pd
import wordninja
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


class TextDataProcessor(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def read_textdata(self):
        with open(self.filepath, "r") as f:
            return pd.read_csv(f)

    def run(self):
        dataframe = self.make_normalization(
            self.drop_examples_with_long_words(
                self.clean_text_as_list(
                    self.clean_text_as_string(
                        self.read_textdata()[["comment_text", "toxic"]]
                    )
                )
            )
        )
        return dataframe

    @staticmethod
    def clean_text_as_string(dataframe):
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

        return dataframe

    @staticmethod
    def clean_text_as_list(dataframe):
        dataframe["comment_text_array"] = (
            dataframe["comment_text_cleaned"]
            .str.split(" ")
            .apply(
                lambda text: remove_stopwords(
                    " ".join(
                        [
                            re.sub(
                                r"(.+?)\1+",
                                r"\1",
                                "".join(
                                    [
                                        c[0]
                                        for c in itertools.groupby(
                                            contractions.fix(word)
                                        )  # remove multiple equal letters
                                    ]
                                ),
                            )  # remove repeating patterns in word
                            for word in text
                            if not "'" in contractions.fix(word)
                        ]  # fix contractions
                    )
                ).split(
                    " "
                )  # remove stopwords ans split words by spaces
            )
        )
        return dataframe

    # wordninja.split(

    @staticmethod
    def make_normalization(dataframe, norm_type="lemma", subtype=None):
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
            ].apply(lambda lst: [stemmer.stem(x) for x in lst])

        elif norm_type == "lemma":
            lemmatizer = WordNetLemmatizer()
            dataframe["norm_type"] = norm_type
            dataframe["comment_text_array_norm"] = dataframe[
                "comment_text_array"
            ].apply(lambda lst: [lemmatizer.lemmatize(x) for x in lst])
        else:
            raise ValueError("Unknown normalizing type: %s" % norm_type)

        return dataframe

    @staticmethod
    def drop_examples_with_long_words(dataframe):
        return dataframe[dataframe["comment_text_array"].apply(is_long_words)]


def is_long_words(lst, large_word_thld=20):
    return not len([len(x) for x in lst if len(x) > large_word_thld]) > 0
