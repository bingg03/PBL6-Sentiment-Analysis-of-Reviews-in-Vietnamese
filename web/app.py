import os
from flask import Flask, render_template, request, jsonify, session, flash
import pandas as pd
from crawl_comments import crawl_comments  # Import hàm crawl dữ liệu
from sentiment_analysis_model import predict_sentiment  # Import hàm phân tích cảm xúc
import random
import json
import re

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'Huuyeuarsenal07&')  # Đặt secret key cho session

# Hàm giả lập phân tích cảm xúc (nếu mô hình chưa có)
def mock_predict_sentiment(comment):
    return random.choice(["Positive", "Negative"])

# Đọc từ trong pos.txt và neg.txt
def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]
positive_words = load_keywords('C:/Users/dell 3400/Documents/PBL6/N1/web/data/pos.txt')
negative_words = load_keywords('C:/Users/dell 3400/Documents/PBL6/N1/web/data/neg.txt')

# Route trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Route crawl bình luận từ URL
@app.route('/crawl', methods=['POST'])
def crawl():
    data = request.get_json()  # Lấy dữ liệu từ JSON
    url = data.get('url')  # Lấy URL từ JSON
    if not url:
        return jsonify({"error": "URL không hợp lệ"}), 400

    try:
        # Gọi hàm crawl để lấy dữ liệu bình luận
        df, file_name = crawl_comments(url)

        if df.empty or 'Comment' not in df.columns:
            return jsonify({"error": "Không có bình luận nào được lấy hoặc dữ liệu thiếu cột 'Comment'"}), 400

        # Lọc và lưu bình luận vào session
        session['comments'] = df['Comment'].dropna().str.strip().tolist()
        return jsonify({"message": "Crawl thành công, đã lưu bình luận."})

    except Exception as e:
        print("Lỗi khi lấy bình luận:", e)
        return jsonify({"error": str(e)}), 500

# Route phân tích cảm xúc
@app.route('/analyze', methods=['POST'])
def analyze():
    comments_list = session.get('comments', [])  # Lấy danh sách bình luận từ session

    if not comments_list:
        return jsonify({"error": "Không có bình luận để phân tích"}), 400

    try:
        # Gọi mô hình BERT
        bert_predictions = predict_sentiment(comments_list, model='bert')
        if isinstance(bert_predictions, list):
            bert_predictions = [pred[0] if isinstance(pred, list) else pred for pred in bert_predictions]
        else:
            bert_predictions = bert_predictions.tolist()  # Chỉ gọi tolist nếu không phải list

        # Gọi mô hình LSTM
        lstm_predictions = predict_sentiment(comments_list, model='lstm')
        if isinstance(lstm_predictions, list):
            lstm_predictions = [pred[0] if isinstance(pred, list) else pred for pred in lstm_predictions]
        else:
            lstm_predictions = lstm_predictions.tolist()  # Chỉ gọi tolist nếu không phải list

        # Hàm phân tích kết quả và highlight
        def process_comments(predictions, model_name):
            positive_count, negative_count = 0, 0
            analyzed_comments = []

            for review, pred_label in zip(comments_list, predictions):
                # Kiểm tra và làm phẳng nếu cần
                if isinstance(pred_label, list):
                    pred_label = pred_label[0]  # Lấy giá trị đầu tiên trong danh sách nếu là list

                # Đảm bảo pred_label là kiểu int
                try:
                    pred_label = int(pred_label)
                except ValueError:
                    pred_label = 0  # Nếu không thể chuyển thành int, gán giá trị mặc định (tiêu cực)

                if pred_label == 1:
                    positive_count += 1
                else:
                    negative_count += 1

                highlighted_text = review
                if pred_label == 1:  # Bình luận tích cực
                    for word in positive_words:
                        highlighted_text = re.sub(
                            rf'\b{re.escape(word)}\b',
                            f"<b><i><u>{word}</u></i></b>",
                            highlighted_text
                        )
                elif pred_label == 0:  # Bình luận tiêu cực
                    for word in negative_words:
                        highlighted_text = re.sub(
                            rf'\b{re.escape(word)}\b',
                            f"<b><i><u>{word}</u></i></b>",
                            highlighted_text
                        )

                analyzed_comments.append({
                    "text": highlighted_text,
                    "sentiment": pred_label,
                })

            total_comments = positive_count + negative_count
            positive_percent = (positive_count / total_comments) * 100 if total_comments > 0 else 0
            negative_percent = (negative_count / total_comments) * 100 if total_comments > 0 else 0

            return {
                "model_name": model_name,
                "comments": analyzed_comments,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_percent": positive_percent,
                "negative_percent": negative_percent,
            }

        # Xử lý kết quả cho từng mô hình
        bert_results = process_comments(bert_predictions, "BERT")
        lstm_results = process_comments(lstm_predictions, "LSTM")

        # Lưu vào session
        session['bert_results'] = bert_results
        session['lstm_results'] = lstm_results

        return jsonify({"message": "Phân tích thành công!", "status": "completed"})

    except Exception as e:
        print("Lỗi khi phân tích cảm xúc:", e)
        return jsonify({"error": str(e)}), 500

# Route hiển thị kết quả
@app.route('/result')
def show_results():
    model = request.args.get('model')  # Lấy mô hình từ tham số URL
    if model == 'bert':
        results = session.get('bert_results', {})
    elif model == 'lstm':
        results = session.get('lstm_results', {})
    else:
        return "Mô hình không hợp lệ hoặc chưa chạy xong.", 400

    return render_template('result.html', result=results)

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True, port=8080)
