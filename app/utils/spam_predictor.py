def predictor(texts, model, vectorizer):
    """
    Fungsi wrapper untuk LIME.
    Menerima list teks dan mengembalikan probabilitas prediksi.
    """
    preprocessed_texts = [preprocess_text(text) for text in texts]
    features = vectorizer.transform(preprocessed_texts)

    predictions = model.predict(features.toarray())

    return predictions