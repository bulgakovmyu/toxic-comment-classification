import spacy
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
import numpy as np
import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
import multiprocessing


class W2VModel(object):
    def __init__(self, classifier_name="lgbm"):
        self.model = self.init_classifier(classifier_name)

    def init_classifier(self, classifier_name):
        if classifier_name == "lgbm":
            return lgb.LGBMClassifier()
        elif classifier_name == "MultNB":
            return MultinomialNB()
        elif classifier_name == "lr":
            return LogisticRegression(solver="lbfgs", max_iter=1000)

    def train_and_validate(self, X_train, y_train, X_test, y_test, f_type="spacy"):
        if f_type == "spacy":
            w2v = spacy.load("en_core_web_md")
            vec_w2v = np.vectorize(lambda x: w2v(str(x)))
            X_train = [list(x.vector) for x in vec_w2v(np.array(X_train))]
            X_test = [list(x.vector) for x in vec_w2v(np.array(X_test))]

        if f_type == "gensim":
            cores = (
                multiprocessing.cpu_count()
            )  # Count the number of cores in a computer
            w2v = Word2Vec(
                size=100,
                min_count=1,
                workers=cores - 1,
            )
            w2v.build_vocab(
                [["I", "am", "a", "sentence"], ["Another", "sentence", "here"]]
            )
            pretrained_path = "/Users/Mikhail_Bulgakov/GitRepo/toxic-comment-classification/models/glove.6B/glove_model2.txt"
            model_buf = KeyedVectors.load_word2vec_format(pretrained_path, binary=False)
            w2v.build_vocab([list(model_buf.vocab.keys())], update=True)
            w2v.intersect_word2vec_format(pretrained_path, binary=False, lockf=1.0)
            w2v.build_vocab([text.split(" ") for text in X_train], update=True)
            w2v.train(
                [text.split(" ") for text in X_train],
                total_examples=w2v.corpus_count,
                epochs=w2v.epochs,
            )

            X_train = [
                np.array([w2v.wv[w] for w in text.split(" ")]).sum(axis=0)
                for text in X_train
            ]
            X_test = [
                np.array(
                    [
                        w2v.wv[w] if catch(lambda: w2v.wv[w]) else np.zeros(100)
                        for w in text.split(" ")
                    ]
                ).sum(axis=0)
                for text in X_test
            ]

        if f_type == "gensim_new":
            cores = (
                multiprocessing.cpu_count()
            )  # Count the number of cores in a computer
            w2v = Word2Vec(
                size=100,
                min_count=1,
                workers=cores - 1,
            )
            w2v.build_vocab(
                [["I", "am", "a", "sentence"], ["Another", "sentence", "here"]]
            )
            w2v.build_vocab([text.split(" ") for text in X_train], update=True)
            w2v.train(
                [text.split(" ") for text in X_train],
                total_examples=w2v.corpus_count,
                epochs=w2v.epochs,
            )

            X_train = [
                np.array([w2v.wv[w] for w in text.split(" ")]).sum(axis=0)
                for text in X_train
            ]
            X_test = [
                np.array(
                    [
                        w2v.wv[w] if catch(lambda: w2v.wv[w]) else np.zeros(100)
                        for w in text.split(" ")
                    ]
                ).sum(axis=0)
                for text in X_test
            ]

        self.model.fit(X_train, y_train)
        train_auc_score = roc_auc_score(
            y_train, [x[1] for x in self.model.predict_proba(X_train)]
        )
        y_pred = self.model.predict(X_test)
        test_auc_score = roc_auc_score(
            y_test, [x[1] for x in self.model.predict_proba(X_test)]
        )
        print("train_auc: {:.5}".format(train_auc_score))
        print("test_auc: {:.5}".format(test_auc_score))
        print(classification_report(y_test, y_pred))
        return self.model, y_pred


def catch(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        return True
    except KeyError:
        return False
