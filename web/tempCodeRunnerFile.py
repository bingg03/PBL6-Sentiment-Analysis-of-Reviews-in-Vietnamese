import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
from crawl_comments import crawl_comments  # Import hàm crawl
from sentiment_analysis_model import predict_sentiment  # Import mô hình phân tích cảm xúc
import random
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'Huuyeuarsenal07&')

# Đường dẫn đến mô hình BERT
BERT_MODEL_PATH = 'models/bert_model.h5'

def mock_predict_sentiment(comment, model):
    # Hàm giả lập cảm xúc: Trả về ngẫu nhiên "Positive", "Negative"
    return random.choice(["Positive", "Negative"])

@app.route('/')
def index():
    return render_template('index.html')

# Route crawl sẽ lấy bình luận từ URL và lưu vào session
@app.route('/crawl', methods=['POST'])
def crawl():
    url = request.form.get('url')
    
    try:
        # Gọi hàm crawl_comments để lấy dữ liệu
        df = crawl_comments(url)
        
        # Kiểm tra nếu DataFrame không trống và có cột 'Comment'
        if df.empty or 'Comment' not in df.columns:
            return jsonify({"error": "Không có bình luận nào được lấy hoặc dữ liệu thiếu cột 'Comment'"}), 400

        # Chuyển DataFrame thành danh sách từ điển và lưu vào session
        session['comments'] = df['Comment'].tolist()
        
        return jsonify({"message": "Crawl thành công, đã lưu bình luận."})

    except Exception as e:
        print("Lỗi khi lấy bình luận:", e)
        return jsonify({"error": str(e)}), 500

# Route analyze sẽ phân tích cảm xúc cho các bình luận đã crawl
@app.route('/analyze', methods=['POST'])
def analyze():
    model = request.form.get('model')  # Nhận giá trị mô hình từ yêu cầu
    
    # Lấy danh sách bình luận từ session
    comments_list = session.get('comments', [])
    if not comments_list:
        return jsonify({"error": "Không có bình luận để phân tích"}), 400
    
    try:
        # Phân tích cảm xúc cho từng bình luận
        analyzed_comments = []
        positive_count = 0
        negative_count = 0

        for text in comments_list:
            if model == 'model_1':
                sentiment = predict_sentiment(text, model='bert')  # Dùng BERT thực sự
            else:
                sentiment = mock_predict_sentiment(text, model)  # Dùng mô hình giả

            if sentiment == "Positive":
                positive_count += 1
            else:
                negative_count += 1

            analyzed_comments.append({"text": text, "sentiment": sentiment})
        
        # Tính phần trăm cảm xúc
        total_comments = positive_count + negative_count
        positive_percent = (positive_count / total_comments) * 100 if total_comments > 0 else 0
        negative_percent = (negative_count / total_comments) * 100 if total_comments > 0 else 0
        
        # Lưu số lượng và tỷ lệ phần trăm vào session
        session['analyzed_comments'] = analyzed_comments
        session['positive_count'] = positive_count
        session['negative_count'] = negative_count
        session['positive_percent'] = positive_percent
        session['negative_percent'] = negative_percent

        return jsonify({"message": "Phân tích thành công!"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route hiển thị kết quả phân tích
@app.route('/result')
def show_results():
    analyzed_comments = session.get('analyzed_comments', [])
    positive_count = session.get('positive_count', 0)
    negative_count = session.get('negative_count', 0)
    
    return render_template(
        'result.html',
        result={
            'comments': analyzed_comments,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    )

if __name__ == '__main__':
    app.run(debug=True)
