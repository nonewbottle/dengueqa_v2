# ======== Ignore Warnings ========
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import warnings

# Save the original stderr
original_stderr = sys.stderr

# Redirect stderr to suppress TensorFlow warnings TEMPORARILY
class DevNull:
    def write(self, msg): pass
    def flush(self): pass

sys.stderr = DevNull()

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# ======== Import Libraries ========
from flask import Flask, render_template, request, session, redirect, url_for
import pickle
import numpy as np
import malaya

# Restore stderr so Flask and others can print to console
sys.stderr = original_stderr

from langdetect import detect
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# === Remove stopwords from a list of tokens ===
malay_stopwords = set(malaya.text.function.get_stopwords())
nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

def remove_stopwords(tokens, lang='en'):
    stopword_set = english_stopwords if lang == 'en' else malay_stopwords
    return [t for t in tokens if t not in stopword_set]

# ========== Detect Rojak ==========
def is_rojak(text):
    tokens = [t for t in word_tokenize(text.lower()) if t.isalpha()]
    malay_stop_count = sum(1 for t in tokens if t in malay_stopwords)
    english_stop_count = sum(1 for t in tokens if t in english_stopwords)
    return malay_stop_count > 0 and english_stop_count > 0

# ========== Malay Preprocessing ==========
tokenizer = malaya.tokenizer.Tokenizer()
stemmer = malaya.stem.sastrawi()

def preprocess_question_malay(q):
    tokens = tokenizer.tokenize(q.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = remove_stopwords(tokens, lang='ms')
    stems = [stemmer.stem(t) for t in tokens]
    return set(stems)

# ========== English Preprocessing ==========
wnl = WordNetLemmatizer()

def preprocess_question_english(q):
    tokens = word_tokenize(q.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = remove_stopwords(tokens, lang='en')
    lemmas = [wnl.lemmatize(t) for t in tokens]
    return set(lemmas)

def jaccard_similarity(a, b):
    return len(a & b) / len(a | b) if a | b else 0

# ========== Flask App ==========
app = Flask(__name__)
app.secret_key = os.urandom(24)

# ========== Load Model and Data ==========
MODEL_PATH = 'test-retrieval_model.pkl'
GOOGLE_DRIVE_FILE_ID = '1Js4Cz8VVMbQLcADESspGEjdlZJVr0N_B'

def download_model():
    if not os.path.exists(MODEL_PATH):
        import gdown
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
        print("Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete!")

download_model()

with open('test-retrieval_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['vectorizer']

# Malay data
question_vecs_malay = data['question_vecs_malay']
questions_malay = data['questions_malay']
answers_malay = data['answers_malay']
q_stems_malay = data['q_stems_malay']

# English data
question_vecs_eng = data['question_vecs_eng']
questions_eng = data['questions_eng']
answers_eng = data['answers_eng']
q_lemma_eng = data['q_lemma_eng']

# ========== Retrieval Function ==========
def retrieve_answer(user_question):
    tokens = [t for t in word_tokenize(user_question.lower()) if t.isalpha()]
    malay_stop_count = sum(1 for t in tokens if t in malay_stopwords)
    english_stop_count = sum(1 for t in tokens if t in english_stopwords)

    # Decide language by which stopword count is higher
    if malay_stop_count > english_stop_count:
        lang = 'ms'
    elif english_stop_count > malay_stop_count:
        lang = 'en'
    else:
        try:
            lang = detect(user_question)
        except Exception:
            lang = 'ms'
    
    if lang in ['ms', 'id']:
        questions = questions_malay
        answers = answers_malay
        question_vecs = question_vecs_malay
        q_stems = q_stems_malay
        user_stems = preprocess_question_malay(user_question)
        not_found_msg = "Maaf, tidak menemui jawapan yang sesuai."
    else:
        questions = questions_eng
        answers = answers_eng
        question_vecs = question_vecs_eng
        q_stems = q_lemma_eng
        user_stems = preprocess_question_english(user_question)
        not_found_msg = "Sorry, could not find a suitable answer."

    # 1. Keyword match (Jaccard)
    jaccard_similarities = [jaccard_similarity(user_stems, stems) for stems in q_stems]
    best_token_idx = np.argmax(jaccard_similarities)
    best_token_score = jaccard_similarities[best_token_idx]

    if best_token_score > 0.5:
        return questions[best_token_idx], answers[best_token_idx], float(best_token_score), "token", lang
    else:
        # 2. Semantic similarity
        user_vec = model.encode([user_question], convert_to_numpy=True)[0]
        similarities = np.dot(question_vecs, user_vec) / (np.linalg.norm(question_vecs, axis=1) * np.linalg.norm(user_vec) + 1e-8)
        best_semantic_idx = np.argmax(similarities)
        best_semantic_score = similarities[best_semantic_idx]
        if best_semantic_score >= 0.6:
            return questions[best_semantic_idx], answers[best_semantic_idx], float(best_semantic_score), "semantic", lang
        else:
            return None, not_found_msg, 0.0, "none", lang

# ========== Routes ==========
@app.route("/", methods=["GET"])
def welcome():
    session.pop("chat_history", None)
    return render_template("welcome.html")

@app.route("/index", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    chat_history = session["chat_history"]
    if request.method == "POST":
        user_question = request.form["question"].strip()
        if user_question:
            retrieved_q, answer, score, method, lang = retrieve_answer(user_question)
            if retrieved_q is None:
                retrieved_q = "(tiada padanan)" if lang in ['ms', 'id', 'rojak'] else "(no match)"
            chat_history.append({
                "user": user_question,
                "bot_q": retrieved_q,
                "bot_a": answer,
                "sim": f"{score:.3f}",
                "method": method,
                "lang": lang
            })
            session["chat_history"] = chat_history
        return redirect(url_for('index'))
    return render_template("index.html", chat_history=chat_history)

@app.route("/reset")
def reset():
    session.pop("chat_history", None)
    return redirect(url_for('index'))

# ========== Main ==========
if __name__ == "__main__":
    app.run(debug=True)