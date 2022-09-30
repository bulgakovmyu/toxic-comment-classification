# %%
from src.preprocess.text import TextDataProcessor

# %%
filepath = "/Users/Mikhail_Bulgakov/GitRepo/toxic-comment-classification/data/jigsaw-toxic-comment-train-processed-seqlen128.csv"
df = TextDataProcessor(filepath=filepath).run()

# %%
df
# %%
