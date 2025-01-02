import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_bert import get_custom_objects, Tokenizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from vncorenlp import VnCoreNLP
from gensim.utils import simple_preprocess
from nltk import flatten
import codecs

# Đường dẫn tới tài nguyên
BERT_MODEL_PATH = "C:/Users/dell 3400/Documents/PBL6/N1/web/models/bert_model.h5"
LSTM_MODEL_PATH = "C:/Users/dell 3400/Documents/PBL6/N1/web/models/lstm_model.h5"
VOCAB_PATH = "C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/voca.txt"
TFIDF_PATH = "C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/tfidf_vectorizer.pkl"
TOKENIZER_PATH = "C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/VnCoreNLP-1.1.1.jar"
SEQ_LEN = 128

# Biến toàn cục
bert_model = None
lstm_model = None
vocab = None
tfidf_vectorizer = None

# Khởi tạo tokenizer cho VnCoreNLP
tokenizer = VnCoreNLP(TOKENIZER_PATH, annotators="wseg", max_heap_size='-Xmx500m')

# Tokenizer cho BERT
with codecs.open("C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/vocab.txt", 'r', 'utf-8') as reader:
    token_dict = {line.strip(): idx for idx, line in enumerate(reader)}
bert_tokenizer = Tokenizer(token_dict, cased=True)

# Hàm tiền xử lý dữ liệu
replace_list = pickle.load(open("C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/replace_list.pkl", 'rb'))
stopwords = [...]  # (Danh sách stopwords như trong code trước)

def preprocess(data):
    token = []
    for text in data:
        if isinstance(text, str):
            text = re.sub(r'([a-z])\1+', lambda m: m.group(1), text.strip().lower())
            text = re.sub(r'<a class.+</a>', ' ', text)
            for k, v in replace_list.items():
                text = text.replace(k, v)
            text = ' '.join(flatten(tokenizer.tokenize(text)))
            tokens = simple_preprocess(text)
            filtered_tokens = [word for word in tokens if word not in stopwords]
            token.append(' '.join(filtered_tokens))
    return token

# Hàm chuẩn hóa dữ liệu đầu vào cho BERT
def load_data(data):
    indices = []
    for text in data:
        ids, segments = bert_tokenizer.encode(text, max_len=SEQ_LEN)
        indices.append(ids)
    return [np.array(indices), np.zeros_like(indices)]

# Hàm chuyển văn bản thành chuỗi số cho LSTM
def text_to_sequence(text, vocab, max_length=25):
    words = text.split()
    sequence = [vocab.get(word, vocab.get('unknown', 2)) for word in words]
    return sequence + [vocab.get('<PAD>', 0)] * (max_length - len(sequence)) if len(sequence) < max_length else sequence[:max_length]

# Hàm tải mô hình LSTM
def load_lstm_model():
    global lstm_model, vocab, tfidf_vectorizer
    if lstm_model is None:
        lstm_model = load_model(LSTM_MODEL_PATH)
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}
        with open(TFIDF_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
    print("Mô hình LSTM và tài nguyên đã được tải.")

# Hàm tải mô hình BERT
def load_bert_model():
    global bert_model
    if bert_model is None:
        bert_model = load_model(BERT_MODEL_PATH, custom_objects=get_custom_objects())
    print("Mô hình BERT đã được tải.")

# Hàm dự đoán với BERT
def predict_sentiment_with_bert(data):
    load_bert_model()
    preprocessed_data = preprocess(data)
    X_test = load_data(preprocessed_data)
    y_pred = bert_model.predict(X_test)
    return np.round(y_pred).flatten()

# Hàm dự đoán với LSTM
def predict_sentiment_with_lstm(data):
    load_lstm_model()
    preprocessed_data = preprocess(data)
    max_length = 25
    X_test = np.array([text_to_sequence(text, vocab, max_length) for text in preprocessed_data])
    y_pred = lstm_model.predict(X_test)
    return (y_pred > 0.5).astype(int).flatten()

# Hàm chính để chọn mô hình và dự đoán
def predict_sentiment(data, model):
    if model == 'bert':
        return predict_sentiment_with_bert(data)
    elif model == 'lstm':
        return predict_sentiment_with_lstm(data)
    else:
        raise ValueError("Model không được hỗ trợ. Chọn 'bert' hoặc 'lstm'.")

# Hàm tính toán các chỉ số đánh giá
def evaluate_model(y_true, y_pred, y_pred_prob=None):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1-Score:", f1_score(y_true, y_pred, zero_division=0))
    if y_pred_prob is not None:
        print("AUC-ROC:", roc_auc_score(y_true, y_pred_prob))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đọc dữ liệu test
    test_data_path = "C:/Users/dell 3400/Documents/PBL6/N1/web/data/test1.csv"
    test_df = pd.read_csv(test_data_path)
    test_reviews = test_df['review']
    y_true = test_df['label'] if 'label' in test_df.columns else None

    # Dự đoán với LSTM
    y_pred = predict_sentiment(test_reviews, model='lstm')

    # Nếu có nhãn, đánh giá mô hình
    if y_true is not None:
        evaluate_model(y_true, y_pred)
