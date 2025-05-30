def predict_and_explain_spam(text, model, vectorizer, explainer, num_features=5):
    """
    Melakukan prediksi SPAM, memberikan penjelasan (rule-based atau LIME),
    dan mengembalikan hasilnya.

    Args:
        text (str): Teks masukan dari pengguna.
        model (keras.Model): Model klasifikasi.
        vectorizer (TfidfVectorizer): Objek TF-IDF.
        explainer (LimeTextExplainer): Objek LIME Explainer.
        num_features (int): Jumlah kata penting yang ingin ditampilkan.

    Returns:
        tuple: (predicted_class_name, prediction_probability, explanation_list, reason_type)
               reason_type bisa 'Rule' atau 'Model'.
    """
    
    # LANGKAH 1: Cek dengan Rule Breaker
    is_spam_by_rule, reason = rule_based_spam_filter(text, threshold=0.3)

    if is_spam_by_rule:
        predicted_class_name = 'SPAM'
        prediction_probability = 1.0  # Menunjukkan keyakinan tinggi karena aturan
        explanation_list = [] # Tidak ada LIME, penjelasan diberikan oleh 'reason'

        # Pesan penjelasan spesifik untuk kasus ini
        if reason == "Terlalu banyak simbol":
            explanation_message = "Pesan terdeteksi SPAM karena: Terlalu banyak simbol pada teks."
            explanation_list = [('Terlalu banyak simbol', 1.0)] # Representasi untuk API

        print(f"\nKalimat yang dimasukkan: {text}")
        print(f"Prediksi: {predicted_class_name} (Berdasarkan Aturan: {reason})")
        print("Pesan ini adalah SPAM 🚫")
        print(f"\nPenjelasan: {explanation_message}")

        return predicted_class_name, prediction_probability, explanation_list, 'Rule'

    # LANGKAH 2: Jika tidak terdeteksi rule, lanjutkan dengan Model & LIME
    preprocessed_text = preprocess_text(text)
    features = vectorizer.transform([preprocessed_text])
    prediction_probs = model.predict(features.toarray())[0]
    predicted_class_index = np.argmax(prediction_probs)
    predicted_class_name = class_names[predicted_class_index]
    prediction_probability = prediction_probs[predicted_class_index]

    print(f"\nKalimat yang dimasukkan: {text}")
    print(f"Prediksi: {predicted_class_name} (Probabilitas Model: {prediction_probability:.2f})")
    if predicted_class_index == 1:
        print("Pesan ini adalah SPAM 🚫")
    else:
        print("Pesan ini BUKAN SPAM ✅")

    # LANGKAH 3: Hasilkan Penjelasan LIME
    explanation = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lambda x: predictor(texts=x, model=model, vectorizer=vectorizer),
        num_features=num_features,
        labels=(predicted_class_index,)
    )

    explanation_list = explanation.as_list(label=predicted_class_index)

    print(f"\nKata-kata yang berkontribusi (terhadap {predicted_class_name} menurut Model):")
    for feature, weight in explanation_list:
        print(f"- {feature}: {weight:.4f}")

    return predicted_class_name, prediction_probability, explanation_list, 'Model'