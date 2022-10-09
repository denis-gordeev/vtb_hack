from typing import Dict

from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
import razdel
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)


class Text(BaseModel):
    text: str


app = FastAPI()

pipe = pipeline("text-classification", "vtb_model")
insight_pipe = pipeline("text-classification", "vtb_insight_model")
tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}


@app.post("/predict")
def predict(text: Text = "") -> Dict:
    """Эндпойнт FastApi для выдачи предсказаний по тексту
    In [22]: requests.post("http://127.0.0.1:8989/predict", json={"tex
    ...: t": "бухгалтер"}).json()
    Out[22]: {'label': 1, 'score': 0.7199110388755798}

    Args:
        text (Text): текст документа

    Returns:
        Dict: словарик с предсказанием
    """
    print(text)
    prediction = pipe(text.text, **tokenizer_kwargs)[0]
    label_str = prediction["label"]
    score = prediction["score"]
    label = int(label_str[-1])
    return {"label": label, "score": score}


@app.post("/predict_insight")
def predict_insight(text: Text = "") -> Dict:
    """Эндпойнт FastApi для выдачи предсказаний по тексту
    In [12]: requests.post(
        "http://127.0.0.1:8989/predict_insight",
        json={"text": "Открыли новый завод в Тьмутаракани"}).json()
    Out[12]: 
    {'insight': 'Открыли новый завод в Тьмутаракани',
    'insight_proba': 0.8225430250167847,
    'insight_location': 'Тьмутаракани'}
    In [16]: requests.post("http://127.0.0.1:8989/predict_insight",
        json={"text": "Медведи любят мёд"}).json()
    Out[16]: {'insight': '', 'insight_proba': 0.015289008617401123,
              'insight_location': ''}

    Args:
        text (Text): текст документа

    Returns:
        Dict: словарик с предсказанием
    """
    insight_proba = 0.0
    insight_sent = ""
    insight_town = ""
    sents = [s.text for s in razdel.sentenize(text.text)]
    for s in sents:
        prediction = insight_pipe(s, **tokenizer_kwargs)[0]
        label_str = prediction["label"]
        score = prediction["score"]
        label = int(label_str[-1])
        if label != 1:
            score = 1 - score
        doc = Doc(s)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        try:
            locations = [
                s[n[0].start : n[0].stop]
                for n in list(doc.ner)[1:]
                if n[0].type == "LOC"
            ]
        except:
            locations = []
        if locations:
            location = locations[0]
        else:
            location = insight_town
        if label == 1 and score > insight_proba:
            insight_sent = s
            insight_proba = score
            insight_town = location
        elif score > insight_proba:
            insight_proba = score
    return {
        "insight": insight_sent,
        "insight_proba": insight_proba,
        "insight_location": insight_town,
    }
