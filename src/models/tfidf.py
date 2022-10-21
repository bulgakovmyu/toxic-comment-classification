from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TFIDFModel(object):
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
        tfidf = TfidfVectorizer(
            max_features=10000,
            analyzer="word",
            ngram_range=(1, 2),
            stop_words="english",
        )
        tfidf.fit(X_train)  # train should be done only with train data
        X_train = tfidf.transform(X_train)
        X_test = tfidf.transform(X_test)

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
