import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import en_core_web_sm
nlp = en_core_web_sm.load()

with open('knn.pkl', 'rb') as file:
    knn = pickle.load(file)

with open('tfidf.pkl', 'rb') as file:
    tfidf = pickle.load(file)

# Fungsi utama Streamlit
def preprocess_text(text):
    # Menghapus karakter khusus dan angka
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\d', '', text)

    # Mengonversi teks ke huruf kecil
    text = text.lower()

    # Menghapus stop words
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    lemmatized_tokens = [nlp(token)[0].lemma_ for token in stemmed_tokens]

    # Menggabungkan kembali token yang sudah diproses
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

def main():
    st.title("Klasifikasi Berita Dengan KNN")

    # Sidebar untuk pengguna memasukkan data
    st.sidebar.header("Masukkan Data")
    input_data = []
    st.sidebar.markdown("Tekanan Darah")
    input_dataTekananDarah = st.sidebar.text_input(f"0 Jika Tekanan Darah Rendah dan 1 Jika Tekanan Darah Tinggi", key = "TekananDarah")


    input_df = pd.DataFrame({"berita": [input_dataTekananDarah]})
    st.subheader("Data yang Dimasukkan")
    st.write(input_df)

    # Melatih model jika tombol ditekan
    if st.sidebar.button("Classification"):

        # Latih model
        data1 = preprocess_text(input_df['berita'])
        data1 = [data1]
        data_normal = tfidf.transform(data1)

        # # Lakukan prediksi
        prediction = knn.predict(data_normal)
        st.write("Hasil Pre-processing:", data1)
        st.subheader("Hasil Prediksi")
        st.write("Prediksi Kelas:", prediction)

if __name__ == "__main__":
    main()