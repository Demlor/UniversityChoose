from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Flask app initialization
app = Flask(__name__)

# File paths for backend and frontend data
file_path_backend = 'TEST.xlsx'  # Backend data Excel file
file_path_frontend = 'Book2.xlsx'  # Frontend data Excel file
file_path_new = 'Sorted_Ngành_Nghề.xlsx'  # Additional Excel file for fields

# Load datasets
data_backend = pd.read_excel(file_path_backend)
data_frontend = pd.read_excel(file_path_frontend)
df_new = pd.read_excel(file_path_new, sheet_name='Sheet1')

# Clean and process bracketed columns
def clean_brackets(column):
    cleaned_data = []
    for entry in column:
        try:
            cleaned_data.append(" ".join(ast.literal_eval(entry)))
        except (ValueError, SyntaxError):
            cleaned_data.append(str(entry).replace("[", "").replace("]", ""))
    return cleaned_data

# Ensure required columns exist, fill missing columns with empty strings
for col in ['Khả năng và Điểm mạnh', 'Sở thích và Đam mê', 'MBTI', 'Tổ hợp môn', 'Ngành nghề']:
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

# Extract options for the frontend
mbti_options = data_frontend['MBTI'].dropna().str.strip().unique().tolist()
subject_combination_options = data_frontend['Tổ hợp môn'].dropna().unique().tolist()

# Extract unique options for fields
unique_fields = df_new['Lĩnh vực'].dropna().unique().tolist()

@app.route('/')
def index():
    # Render the HTML template with data
    return render_template(
        'index.html',
        mbti_options=mbti_options,
        subject_combination_options=subject_combination_options,
        fields=unique_fields
    )

@app.route('/recommend-career', methods=['POST'])
def recommend_career():
    user_data = request.json
    mbti = user_data.get('mbti', '').strip().upper()
    subjects = user_data.get('subjects', [])
    subjects = [subject.strip().upper() for subject in subjects]
    strengths = set(user_data.get('strengths', []))  # Chuyển sang tập hợp để dễ so sánh
    interests = set(user_data.get('interests', []))  # Chuyển sang tập hợp để dễ so sánh

    # Tính toán vector hóa cho từng yếu tố
    mbti_vector = tfidf_vectorizer.transform([mbti]) if mbti else None
    subjects_vector = tfidf_vectorizer.transform([" ".join(subjects)]) if subjects else None

    suggestions = []
    for i in range(len(data_backend)):
        row = data_backend.iloc[i]
        row_combined = row['Combined']

        # Kiểm tra MBTI
        if mbti and mbti == row['MBTI'].strip().upper():
            mbti_score = 1.0
        else:
            mbti_score = cosine_similarity(mbti_vector, tfidf_vectorizer.transform([row['MBTI']])).flatten()[0] if mbti_vector is not None else 0

        # Kiểm tra Tổ hợp môn
        if subjects and any(subject in row['Tổ hợp môn'].upper() for subject in subjects):
            subjects_score = 1.0
        else:
            subjects_score = cosine_similarity(subjects_vector, tfidf_vectorizer.transform([row['Tổ hợp môn']])).flatten()[0] if subjects_vector is not None else 0

        # Kiểm tra tỷ lệ trùng khớp cho Khả năng và Điểm mạnh
        backend_strengths = set(row['Khả năng và Điểm mạnh'].split())
        match_strengths = len(strengths.intersection(backend_strengths)) / len(backend_strengths) if backend_strengths else 0
        if match_strengths > 0.75:
            strengths_score = 1.0
        elif 0.6 <= match_strengths <= 0.75:
            strengths_score = 0.75
        elif 0.4 <= match_strengths < 0.6:
            strengths_score = 0.5
        else:
            strengths_score = cosine_similarity(tfidf_vectorizer.transform([" ".join(strengths)]), tfidf_vectorizer.transform([row['Khả năng và Điểm mạnh']])).flatten()[0]

        # Kiểm tra tỷ lệ trùng khớp cho Sở thích và Đam mê
        backend_interests = set(row['Sở thích và Đam mê'].split())
        match_interests = len(interests.intersection(backend_interests)) / len(backend_interests) if backend_interests else 0
        if match_interests > 0.75:
            interests_score = 1.0
        elif 0.6 <= match_interests <= 0.75:
            interests_score = 0.75
        elif 0.4 <= match_interests < 0.6:
            interests_score = 0.5
        else:
            interests_score = cosine_similarity(tfidf_vectorizer.transform([" ".join(interests)]), tfidf_vectorizer.transform([row['Sở thích và Đam mê']])).flatten()[0]

        # Tính điểm trung bình với trọng số
        final_score = (
            mbti_score * 0.2 +
            subjects_score * 0.3 +
            strengths_score * 0.3 +
            interests_score * 0.2
        )

        # Lưu kết quả ngành nghề và điểm trung bình
        suggestions.append((row['Ngành nghề'], final_score))

    # Sắp xếp và lấy top 10 gợi ý
    suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)[:10]

    return jsonify([{"career": s[0], "score": f"{s[1]*100:.2f}%"} for s in suggestions])

@app.route('/submit', methods=['POST'])
def submit():
    family_advice = request.form.get('family_advice')
    financial_influence = request.form.get('financial_influence')
    family_industry = request.form.get('family_industry')

    # Handle the submission and process the data
    return jsonify({
        'family_advice': family_advice,
        'financial_influence': financial_influence,
        'family_industry': family_industry
    })

if __name__ == '__main__':
    app.run(debug=True)
