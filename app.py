import streamlit as st
import torch
import torch.nn.functional as F
import shap
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- SETTINGS ----------------
MODEL_PATH = "bert_fake_news_model"   # folder where model + tokenizer saved
MAX_LEN = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- PREDICT PROB FOR SHAP ----------------
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]

    texts = [str(t) for t in texts]

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        probs = F.softmax(outputs.logits, dim=1)

    return probs.cpu().numpy()

# ---------------- SHAP EXPLAINER ----------------
@st.cache_resource
def load_explainer():
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        predict_proba,
        masker,
        output_names=["FAKE", "REAL"]
    )
    return explainer

explainer = load_explainer()

# ---------------- SIMPLE PREDICTION ----------------
def predict(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        probs = F.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return pred, confidence

# ---------------- UI ----------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("📰 Fake News Detector with Explainable AI")
st.write("Enter a news article or headline to classify and see why the model predicted it.")

text = st.text_area("News text", height=200)

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # ---------- prediction ----------
        pred, conf = predict(text)

        label_map = {0: "FAKE", 1: "REAL"}
        result = label_map[pred]

        st.subheader("Prediction")
        if result == "FAKE":
            st.error(result)
        else:
            st.success(result)

        st.write(f"Confidence: {conf:.3f}")
        st.progress(float(conf))

        # ---------- SHAP explanation ----------
        st.subheader("Model Explanation")

        with st.spinner("Generating explanation..."):
            shap_values = explainer([text])

        # side by side explanation
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔴 Class 0 — FAKE reasoning")
            fake_html = shap.plots.text(shap_values[0, :, 0], display=False)
            st.components.v1.html(fake_html, height=320, scrolling=True)

        with col2:
            st.markdown("### 🔵 Class 1 — REAL reasoning")
            real_html = shap.plots.text(shap_values[0, :, 1], display=False)
            st.components.v1.html(real_html, height=320, scrolling=True)