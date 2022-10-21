import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from src.preprocess.text import TextDataProcessor, DataSet
from sklearn.linear_model import LogisticRegression

import numpy as np
import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score


DEVICE = "cpu"


class BertModel(object):
    def __init__(self, classifier_name="lgbm"):
        self.model = self.init_classifier(classifier_name)

    def init_classifier(self, classifier_name):
        if classifier_name == "lgbm":
            return lgb.LGBMClassifier()
        elif classifier_name == "MultNB":
            return MultinomialNB()
        elif classifier_name == "lr":
            return LogisticRegression(solver="lbfgs", max_iter=1000)

    def train_and_validate(self, X_train, y_train, X_test, y_test):

        X_train = [np.array(embs) for embs in X_train]
        X_test = [np.array(embs) for embs in X_test]

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
