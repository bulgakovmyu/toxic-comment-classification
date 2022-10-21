# %%
from time import time
from src.preprocess.text import TextDataProcessor, DataSet
from src.models.tfidf import TFIDFModel
from src.models.word2vec import W2VModel
from src.models.fasttext import FastTextModel
from src.models.doc2vec import Doc2VecModel

# %%
filepath = "/Users/Mikhail_Bulgakov/GitRepo/toxic-comment-classification/data/jigsaw-toxic-comment-train-processed-seqlen128.csv"
df = TextDataProcessor(filepath=filepath).run(
    norm_type="lemma", simple_preprocess=False, load_ready=True
)

# %%
dataset = DataSet(df)
# %%
start = time()
tfidf_model = TFIDFModel(classifier_name="lr")
tfidf_model.train_and_validate(
    dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
)
print(time() - start)
# %%
start = time()
w2v_model = W2VModel(classifier_name="lr")
w2v_model.train_and_validate(
    dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test, f_type="spacy"
)
print(time() - start)
# %%
start = time()
w2v_model = W2VModel(classifier_name="lr")
w2v_model.train_and_validate(
    dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test, f_type="gensim"
)
print(time() - start)
# %%
start = time()
w2v_model = W2VModel(classifier_name="lr")
w2v_model.train_and_validate(
    dataset.X_train,
    dataset.y_train,
    dataset.X_test,
    dataset.y_test,
    f_type="gensim_new",
)
print(time() - start)
# %%
start = time()
ft_model = FastTextModel(classifier_name="lr")
ft_model.train_and_validate(
    dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
)
print(time() - start)
# %%
start = time()
ft_model = Doc2VecModel(classifier_name="lr")
ft_model.train_and_validate(
    dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
)
print(time() - start)

# %%
