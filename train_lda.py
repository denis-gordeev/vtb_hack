import os
from functools import lru_cache

import pandas as pd
import pymorphy2
import razdel
from topic_model import TopicModelWrapperARTM
from nltk.corpus import stopwords

BOOKKEEPER_ONLY = True
df = pd.read_csv("bookkeeper.csv")

if BOOKKEEPER_ONLY:
    df = df[df["category"] == "bookkeeper"]
    category = "bookkeeper"
else:
    category = "all"

df = df.drop_duplicates(subset=["summary"])
df = df[~df["summary"].isna()]
df = df[["summary", "published", "category"]]
df.columns = ["text", "date", "topics"]

df["date"] = pd.to_datetime(df["date"], utc=True)

morph = pymorphy2.MorphAnalyzer()
stopwords = stopwords.words("russian")
stopwords = set(stopwords)
stopwords.update({"какой", "мочь", "ещё", "кто", "смочь", "это", "такой"})

@lru_cache()
def lemmatize(word):
    return morph.parse(word)[0].normal_form


lemmed_stopwords = {lemmatize(w) for w in stopwords}


def process_text(text):
    words = [w.text for w in razdel.tokenize(text)]
    words = [w for w in words if len(w) >= 3]
    words = [
        w for w in words if w not in lemmed_stopwords and w not in stopwords
    ]
    lemmas = [lemmatize(w).lower() for w in words]
    lemmas = [
        w for w in lemmas if w not in lemmed_stopwords and w not in stopwords
    ]
    return lemmas


# artm.BatchVectorizer crashes if finds ':' in text
tokenized = [process_text(t.replace(":", "")) for t in df["text"].values]

n_topics = 5
os.makedirs(f"bookkeeper_{n_topics}/data/ready2viz/{category}/", exist_ok=True)
topic_model = TopicModelWrapperARTM(
    dir_path=f"lda_{n_topics}_{category}",
    name_dataset=f"bookk{n_topics}",
    n_topics=n_topics,
)
topic_model.prepare_data(list(tokenized))
topic_model.fit()
topic_model.save_top_words(
    f"bookkeeper_{n_topics}/data/ready2viz/tw_{category}.json"
)
theta = topic_model.transform()
df = df.reset_index()
result = theta.merge(
    df[["date"]],
    left_index=True,
    right_index=True,
)
result.to_csv(
    f"bookkeeper_{n_topics}/data/ready2viz/{category}/{category}.gzip.csv",
    compression="gzip",
    index=False,
)
