
import streamlit as st
import joblib

st.title("🔍 Mental Health Sentiment App")

# Load model dan vectorizer
svm_model = joblib.load("model/svm_model.pkl")
xgb_model = joblib.load("model/xgb_model.pkl")
vectorizer = joblib.load("vectorizer/tfidf_vectorizer.pkl")

st.title("Analisis Sentimen Konten Kesehatan Mental TikTok")
st.markdown("Model: Support Vector Machine & XGBoost")

text_input = st.text_area("Masukkan komentar TikTok:")

model_choice = st.selectbox("Pilih Model:", ["SVM", "XGBoost"])

if st.button("Prediksi"):
    if text_input.strip() != "":
        text_vec = vectorizer.transform([text_input])
        if model_choice == "SVM":
            prediction = svm_model.predict(text_vec)[0]
        else:
            prediction = xgb_model.predict(text_vec)[0]

        label = {0: "Negatif", 1: "Netral", 2: "Positif"}
        st.success(f"Hasil Sentimen: **{label.get(prediction, prediction)}**")
    else:
        st.warning("Masukkan komentar terlebih dahulu.")
