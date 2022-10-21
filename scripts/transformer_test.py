# %%
from time import time
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from src.preprocess.text import TextDataProcessor, DataSet
import json
import numpy as np
from src.models.bert import BertModel

# %%

with open("../data/test_embeddings_from_bert.json", "r") as file:
    X_test = json.load(file)

# %%
with open("../data/train_embeddings_from_bert.json", "r") as file:
    X_train = json.load(file)
# %%
filepath = "/Users/Mikhail_Bulgakov/GitRepo/toxic-comment-classification/data/jigsaw-toxic-comment-train-processed-seqlen128.csv"
df = TextDataProcessor(filepath=filepath).run(
    norm_type="lemma", simple_preprocess=False, load_ready=True, for_bert=True
)
# %%
X_train = [np.array(embs) for embs in X_train]
X_test = [np.array(embs) for embs in X_test]
# %%
dataset = DataSet(df)

y_train = dataset.y_train
y_test = dataset.y_test
del dataset
del df
# %%
print(len(X_test))
print(len(y_test))
print(len(X_train))
print(len(y_train))
# %%
start = time()
tfidf_model = BertModel(classifier_name="lr")
tfidf_model.train_and_validate(X_train, y_train, X_test, y_test)
print(time() - start)
# %%
