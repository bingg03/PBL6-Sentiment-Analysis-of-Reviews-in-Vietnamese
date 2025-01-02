import numpy as np
import tensorflow as tf
import random
import re
import os
import pandas as pd
import pickle

from tensorflow.keras.models import model_from_json, load_model
from keras_bert import get_custom_objects

import seaborn as sns
import h5py

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization,
    Bidirectional, Input, Layer, Masking, concatenate
)
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, f1_score)

# Khởi tạo lại cấu trúc mô hình
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=25),  # Điều chỉnh tham số khớp với mô hình gốc
    LSTM(128, return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Đường dẫn tới mô hình BERT
BERT_MODEL_PATH ='C:/Users/dell 3400/Documents/PBL6/N1/web/models/bert_model.h5'

# Đường dẫn tới mô hình LSTM
# Đường dẫn tới mô hình và tài nguyên
LSTM_MODEL_PATH = "C:/Users/dell 3400/Documents/PBL6/N1/web/models/lstm_model.h5"
VOCAB_PATH = "C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/voca.txt"

# Biến toàn cục để lưu mô hình BERT sau khi load
bert_model = None

# Biến toàn cục để lưu mô hình LSTM sau khi load
lstm_model = None
vocab = None

from vncorenlp import VnCoreNLP
tokenizer = VnCoreNLP("C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
import pickle
replace_list = pickle.load(open('C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/replace_list.pkl','rb'))
import re
from gensim.utils import simple_preprocess
import pandas as pd
from nltk import flatten

VN_CHARS_LOWER = u'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð'
VN_CHARS_UPPER = u'ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸÐĐ'
VN_CHARS = VN_CHARS_LOWER + VN_CHARS_UPPER
def no_marks(s):
    __INTAB = [ch for ch in VN_CHARS]
    __OUTTAB = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"*2
    __OUTTAB += "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"*2
    __r = re.compile("|".join(__INTAB))
    __replaces_dict = dict(zip(__INTAB, __OUTTAB))
    result = __r.sub(lambda m: __replaces_dict[m.group(0)], s)
    return result

# Load LSTM


def load_lstm_model():
    global lstm_model, vocab, tfidf_vectorizer
    if lstm_model is None:
        lstm_model = load_model(LSTM_MODEL_PATH)
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            vocab = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f}
    print("Mô hình LSTM và tài nguyên đã được tải.")

#LOAD BERT

def load_bert_model():
    """
    Hàm để tải mô hình BERT từ tệp .h5
    """
    global bert_model
    if bert_model is None:
        try:
            bert_model = load_model(BERT_MODEL_PATH, custom_objects=get_custom_objects())
            print("Mô hình BERT đã được tải thành công!")
        except Exception as e:
            print(f"Lỗi khi tải mô hình BERT: {e}")

stopwords = [
    "anh", "chị", "bạn", "mình", "tôi", "ta", "ông", "họ", "em", "nó",
    "chúng_ta", "chúng_tôi", "trên", "dưới", "trong", "ngoài", "giữa",
    "của", "với", "tại", "đến", "qua", "vào", "bên", "và", "nhưng",
    "hoặc", "mà", "vì", "bởi", "tuy_nhiên", "nếu", "rồi", "sau",
    "sau_đó", "do", "tất_cả", "mỗi", "rất", "nhiều", "ít", "vài",
    "hơn", "hết", "cả", "tất", "bây_giờ", "hôm_nay", "ngày", "đêm",
    "trước", "hiện", "lúc", "khi", "đã", "vừa", "cái", "này", "kia",
    "gì", "điều", "việc", "vậy", "thế", "là", "có", "được", "sẽ",
    "làm", "như", "sao", "tại_sao", "thế_nào", "như_vậy", "vậy_nên",
    "vậy_mà", "vậy_thì", "phải", "đấy", "đây"
]

def text_to_sequence(text, vocab, max_length=25):
    """
    Chuyển đổi văn bản sang chuỗi số dựa trên từ điển.
    """
    words = text.split()
    sequence = []
    for word in words:
        if word.isdigit():
            sequence.append(vocab.get('digit', 1))  # Mặc định mã hóa số là 1
        elif word in vocab:
            sequence.append(vocab[word])
        else:
            sequence.append(vocab.get('unknown', 2))  # Mặc định mã hóa từ không biết là 2
    # Điều chỉnh độ dài của chuỗi
    sequence = sequence[:max_length]  # Cắt chuỗi nếu vượt quá max_length
    sequence += [vocab.get('<PAD>', 0)] * (max_length - len(sequence))  # Thêm <PAD> để đạt max_length
    return sequence


def preprocess(data):
    token = []
    for text in data:
        check = re.search(r'([a-z])\1+',text)
        if check:
          if len(check.group())>2:
            text = re.sub(r'([a-z])\1+', lambda m: m.group(1), text, flags=re.IGNORECASE) #remove các ký tự kéo dài như hayyy,ngonnnn...

        text = text.strip() #loại dấu cách đầu câu
        text = text.lower() #chuyển tất cả thành chữ thường

        text = re.sub('< a class.+</a>',' ',text)

        for k, v in replace_list.items():       #replace các từ có trong replace_list
          text = text.replace(k, v)

        text = re.sub(r'_',' ',text)

        text = ' '.join(i for i in flatten(tokenizer.tokenize(text)))             #gán từ ghép

        tokens = simple_preprocess(text)
        filtered_tokens = [word for word in tokens if word not in stopwords]
        text = ' '.join(filtered_tokens)

        token.append(text)
    return token

import os
import codecs
import numpy as np
import pandas as pd
import tensorflow as tf
token_dict = {}
with codecs.open('C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/vocab.txt', 'rb','utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
from keras_bert import Tokenizer
tokenizer = Tokenizer(token_dict,cased=True)
SEQ_LEN = 128
from sklearn.model_selection import train_test_split
def load_data(data):
    global tokenizer
    indices = []
    for text in data:
        ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
        indices.append(ids)

    indices = np.array(indices)  # Chuyển đổi thành numpy array
    return [indices, np.zeros_like(indices)]


def predict_sentiment_with_bert(data):
    # Đảm bảo mô hình đã được load
    load_bert_model()

    review_data = [str(review) for review in data if isinstance(review, str) or pd.notna(review)]
    data_test = preprocess(review_data)
    
    X_test = load_data(data_test)
    y_pred = np.round(bert_model.predict(X_test))

    return y_pred

def predict_sentiment_with_lstm(data):
    """
    Phân tích cảm xúc bằng mô hình LSTM.
    """
    # Đảm bảo mô hình và từ điển đã được load
    load_lstm_model()

    # Tiền xử lý dữ liệu
    review_data = [str(review) for review in data if isinstance(review, str) or pd.notna(review)]
    data_test = preprocess(review_data)

    max_length = 25
    X_test = np.array([text_to_sequence(text, vocab, max_length) for text in data_test])

    # Dự đoán nhãn
    predictions = lstm_model.predict(X_test)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    return predicted_labels

def mock_predict_sentiment(text, model):
    """
    Hàm mô phỏng phân tích cảm xúc với giá trị ngẫu nhiên cho mô hình 2
    """
    sentiments = ["Positive", "Negative"]
    return random.choice(sentiments)

def predict_sentiment(data, model='bert'):
    """
    Hàm chính để phân tích cảm xúc, chọn mô hình phù hợp
    """
    if model == 'bert':
        return predict_sentiment_with_bert(data)
    elif model == 'lstm':
        return predict_sentiment_with_lstm(data)
        # return [mock_predict_sentiment(text, model='lstm') for text in data]
    else:
        raise ValueError("Model không được hỗ trợ. Chọn 'bert' hoặc 'lstm'.")