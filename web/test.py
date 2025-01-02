TFIDF_PATH = "C:/Users/dell 3400/Documents/PBL6/N1/web/vncorenlp/tfidf_vectorizer.pkl"

import os
if os.path.exists(TFIDF_PATH):
    print("File tồn tại.")
else:
    print("File không tồn tại. Vui lòng kiểm tra đường dẫn.")

import pickle

try:
    with open(TFIDF_PATH, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
        print("TF-IDF Vectorizer đã được tải thành công!")
        print("Từ điển TF-IDF:", tfidf_vectorizer.vocabulary_)  # Kiểm tra từ điển
except Exception as e:
    print(f"Lỗi khi tải TF-IDF Vectorizer: {e}")