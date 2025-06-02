from fastapi import FastAPI, Request
from pydantic import BaseModel
from keras.models import load_model
from lime.lime_text import LimeTextExplainer
import pickle
import numpy as np
from keras.models import load_model
from lime.lime_text import LimeTextExplainer
from utils.spam_preprocess_text import cleaning, casefolding, handle_slangwords, tokenizing, remove_stopwords, text_result, preprocess_text
from utils.spam_predictor import predictor
from utils.spam_rule_based_filter import rule_based_spam_filter
from utils.spam_predict_and_explain import predict_and_explain_spam

# Inisialisasi FastAPI
app = FastAPI()

# Load model & tools
model = load_model("model/model_spam.h5")
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
explainer = LimeTextExplainer(class_names=['Not SPAM', 'SPAM'])

# Define input schema
class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: TextInput):
    text = input_data.text
    pred_class, prob, explanation, source = predict_and_explain_spam(
        text=text,
        model=model,
        vectorizer=vectorizer,
        explainer=explainer,
        class_names=['Not SPAM', 'SPAM']
    )
    return {
        "prediction": pred_class,
        "probability": round(prob, 4),
        "explanation": explanation,
        "source": source
    }
