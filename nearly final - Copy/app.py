from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

# Flask app initialization
app = Flask(__name__)

# File paths
file_path_backend = './data/Sorted_Ngành_Nghề.xlsx'  # Backend data Excel file
file_path_frontend = './data/Book1.xlsx'  # Frontend data Excel file

# Load datasets
data_backend = pd.read_excel(file_path_backend)
data_frontend = pd.read_excel(file_path_frontend)

# Function to clean and process bracketed columns
def clean_brackets(column):
    cleaned_data = []
    for entry in column:
        try:
            cleaned_data.append(" ".join(ast.literal_eval(entry)))
        except (ValueError, SyntaxError):
            cleaned_data.append(str(entry).replace("[", "").replace("]", ""))
    return cleaned_data

# Ensure required columns exist, fill missing columns with empty strings
for col in ['Khả năng và Điểm mạnh', 'Sở thích và Đam mê', 'MBTI', 'Tổ hợp môn']:
    if col not in data_backend.columns:
        print(f"Warning: Column '{col}' is missing in the backend data. Adding empty placeholder.")
        data_backend[col] = ""

# Clean specific columns
data_backend['Khả năng và Điểm mạnh'] = clean_brackets(data_backend['Khả năng và Điểm mạnh'])
data_backend['Sở thích và Đam mê'] = clean_brackets(data_backend['Sở thích và Đam mê'])

# Combine relevant columns for TF-IDF processing
data_backend['Combined'] = (
    data_backend['MBTI'].fillna("") + " " +
    data_backend['Tổ hợp môn'].fillna("") + " " +
    data_backend['Khả năng và Điểm mạnh'].fillna("") + " " +
    data_backend['Sở thích và Đam mê'].fillna("")
)

# Vectorize the combined data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data_backend['Combined'])

@app.route('/submit', methods=['POST'])
def submit():
    # Lấy dữ liệu từ frontend (form HTML)
    data = request.get_json()

    # Dữ liệu từ form
    family_advice = data.get('family_advice', '')
    financial_influence = data.get('financial_influence', '')
    family_industry = data.get('family_industry', '')
    field = data.get('field', '')

    # Lấy dữ liệu từ các lựa chọn khác (MBTI, tổ hợp môn, khả năng, sở thích)
    mbti = data.get('mbti', '').strip().upper()
    subjects = data.get('subjects', [])
    subjects = [subject.strip().upper() for subject in subjects]
    strengths = set(data.get('strengths', []))
    interests = set(data.get('interests', []))

    # Vectorize user inputs
    strengths_vector = tfidf_vectorizer.transform([" ".join(strengths)]) if strengths else None
    interests_vector = tfidf_vectorizer.transform([" ".join(interests)]) if interests else None

    suggestions = []
    for i in range(len(data_backend)):
        row = data_backend.iloc[i]

        # MBTI - kiểm tra nếu bất kỳ giá trị nào khớp
        backend_mbtis = {mbti.strip() for mbti in str(row['MBTI']).replace('\t', '').split(',')}
        if mbti in backend_mbtis:
            mbti_score = 1.0
        else:
            mbti_score = 0.0

        # Tổ hợp môn - kiểm tra nếu bất kỳ giá trị nào khớp
        backend_subjects = {subject.strip() for subject in str(row['Tổ hợp môn']).replace('\t', '').split(',')}
        if any(subject in backend_subjects for subject in subjects):
            subjects_score = 1.0
        else:
            subjects_score = 0.0

        # MAIN STRENGTHS và phần còn lại của Khả năng và Điểm mạnh
        if row['MAIN STRENGTHS']:
            main_strengths_score = cosine_similarity(
                strengths_vector,
                tfidf_vectorizer.transform([row['MAIN STRENGTHS']])
            ).flatten()[0] if strengths_vector is not None else 0
        else:
            main_strengths_score = 0

        if row['Khả năng và Điểm mạnh']:
            remaining_strengths_score = cosine_similarity(
                strengths_vector,
                tfidf_vectorizer.transform([row['Khả năng và Điểm mạnh']])
            ).flatten()[0] if strengths_vector is not None else 0
        else:
            remaining_strengths_score = 0

        # Tổng điểm cho Khả năng và Điểm mạnh
        strengths_score = main_strengths_score * 0.35 + remaining_strengths_score * 0.65

        # MAIN INTERESTEDS và phần còn lại của Sở thích và Đam mê
        if row['MAIN INTERESTEDS']:
            main_interests_score = cosine_similarity(
                interests_vector,
                tfidf_vectorizer.transform([row['MAIN INTERESTEDS']])
            ).flatten()[0] if interests_vector is not None else 0
        else:
            main_interests_score = 0

        if row['Sở thích và Đam mê']:
            remaining_interests_score = cosine_similarity(
                interests_vector,
                tfidf_vectorizer.transform([row['Sở thích và Đam mê']])
            ).flatten()[0] if interests_vector is not None else 0
        else:
            remaining_interests_score = 0

        # Tổng điểm cho Sở thích và Đam mê
        interests_score = main_interests_score * 0.35 + remaining_interests_score * 0.65

        # Tính điểm trung bình với trọng số
        PF_score = (
            mbti_score * 0.2 +
            subjects_score * 0.3 +
            strengths_score * 0.3 +
            interests_score * 0.2
        )

        # Lưu kết quả ngành nghề và điểm trung bình
        suggestions.append((row['Ngành'], PF_score))

    # Sắp xếp và lấy top 10 gợi ý
    suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)[:10]

    return jsonify([{"career": s[0], "score": f"{s[1]*100:.2f}%"} for s in suggestions])


@app.route('/')
def index():
    # Sắp xếp dữ liệu trước khi gửi đến giao diện
    mbti_options = sorted(data_frontend['MBTI'].dropna().unique())
    subject_combination_options = sorted(data_frontend['Tổ hợp môn'].dropna().unique())
    strengths_options = sorted(data_frontend['Khả năng và Điểm mạnh'].dropna().unique())
    interests_options = sorted(data_frontend['Sở thích và Đam mê'].dropna().unique())
    field_options = sorted(data_frontend['Lĩnh vực'].dropna().unique())

    return render_template(
        'index.html',
        mbti_options=mbti_options,
        subject_combination_options=subject_combination_options,
        strengths_options=strengths_options,
        interests_options=interests_options,
        field_options=field_options
    )

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
