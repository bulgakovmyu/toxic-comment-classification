# %%
from src.preprocess.text import TextDataProcessor, DataSet
from time import time
import regex
import contractions

# %%
start = time()
filepath = "/Users/Mikhail_Bulgakov/GitRepo/toxic-comment-classification/data/jigsaw-toxic-comment-train-processed-seqlen128.csv"
df = TextDataProcessor(filepath=filepath).run(
    norm_type="lemma",
    simple_preprocess=False,
    for_bert=True,
    load_ready=False,
    save=True,
)
print(time() - start)
# %%
df

# %%
from itertools import chain
from nltk.probability import FreqDist

tokens = list(chain.from_iterable(df["comment_text_array_norm"]))
# %%
fd_wct = FreqDist(tokens)
print(len(fd_wct))
print(len([(word, fd_wct[word]) for word in fd_wct.keys() if fd_wct[word] > 3]))
print(len([(word, fd_wct[word]) for word in fd_wct.keys() if fd_wct[word] > 10]))
print(len([(word, fd_wct[word]) for word in fd_wct.keys() if fd_wct[word] > 100]))
print(len([(word, fd_wct[word]) for word in fd_wct.keys() if fd_wct[word] > 1000]))
# %%
fd_wct.most_common(10)
# %%
rare_wds = list(fd_wct.keys())[-100:]
[(word, fd_wct[word]) for word in rare_wds]
# %%
df[df.toxic == 1]
# %%
dataset = DataSet(dataframe=df)
# %%
dataset.X_train
# %%
df["comment_text_array_norm"].to_list()
# %%
import spacy

w2v = spacy.load("en_core_web_md")
# %%
test = df["comment_text_array_norm"].iloc[:2].parallel_apply(w2v)
# %%
test.iloc[0].vector
# %%
array = df["comment_text_array_norm"].iloc[:2].to_numpy()
array
# %%
import numpy as np

vec_w2v = np.vectorize(lambda x: w2v(str(x)).vector.tolist())
# %%
vec_w2v(array)
# %%
[list(x.vector) for x in vec_w2v(array)]
# %%
from time import time
from src.preprocess.text import TextDataProcessor, DataSet
from src.models.tfidf import TFIDFModel
from src.models.word2vec import W2VModel
from src.models.fasttext import FastTextModel
from src.models.doc2vec import Doc2VecModel

# %%
filepath = "/Users/Mikhail_Bulgakov/GitRepo/toxic-comment-classification/data/jigsaw-toxic-comment-train-processed-seqlen128.csv"
df = TextDataProcessor(filepath=filepath).run(norm_type="lemma", load_ready=True)

# %%
dataset = DataSet(df)
# %%
start = time()
ft_model = Doc2VecModel(classifier_name="lgbm")
ft_model.train_and_validate(
    dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
)
print(time() - start)
# %%
from gensim.parsing.preprocessing import remove_stopwords

remove_stopwords("hi thanks for our kind words see you around talk")
# %%
