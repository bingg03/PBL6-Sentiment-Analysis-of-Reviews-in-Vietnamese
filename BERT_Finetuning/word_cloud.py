import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv('./data_training/data_cleaned_no_sw.csv', encoding='utf-8-sig')

# Bước 2: Tách dữ liệu thành hai nhóm dựa trên giá trị của cột 'label'
df_label_0 = df[df['label'] == 0]
df_label_1 = df[df['label'] == 1]

# Bước 3: Kết hợp các review thành văn bản lớn cho mỗi nhóm
text_label_0 = ' '.join(df_label_0['review'].astype(str).tolist())
text_label_1 = ' '.join(df_label_1['review'].astype(str).tolist())

# Bước 4: Tạo word cloud cho mỗi nhóm
wordcloud_label_0 = WordCloud(width=1000, height=600, background_color='white').generate(text_label_0)
wordcloud_label_1 = WordCloud(width=1000, height=600, background_color='white').generate(text_label_1)

# Hiển thị word cloud
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.title('Word Cloud cho label 0')
plt.imshow(wordcloud_label_0, interpolation='bilinear')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Word Cloud cho label 1')
plt.imshow(wordcloud_label_1, interpolation='bilinear')
plt.axis('off')

plt.show()
