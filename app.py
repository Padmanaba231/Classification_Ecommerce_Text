import gradio as gr
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load artefak
tfidf = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('svm_model.pkl')
le    = joblib.load('label_encoder.pkl')

stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def classify(text: str):
    cleaned  = preprocess_text(text)
    features = tfidf.transform([cleaned])
    pred_idx = model.predict(features)[0]
    label    = le.inverse_transform([pred_idx])[0]
    return label

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=5, placeholder="Masukkan deskripsi produk..."),
    outputs=gr.Label(label="Kategori Produk"),
    title="🛒 E-Commerce Product Classifier",
    description="Klasifikasi teks produk e-commerce menggunakan TF-IDF + SVM (LinearSVC).",
    examples=[
        ["Men's slim fit cotton casual shirt with button-down collar"],
        ["Wireless bluetooth headphones with noise cancellation"],
        ["Non-stick frying pan with heat resistant handle"],
        ["Python programming book for beginners with exercises"],
    ]
)

demo.launch()