from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Đọc dữ liệu từ file Excel và lấy danh sách các lĩnh vực
file_path_new = 'Sorted_Ngành_Nghề.xlsx'
df_new = pd.read_excel(file_path_new, sheet_name='Sheet1')

# Lấy các giá trị duy nhất từ cột "Lĩnh vực"
unique_fields = df_new['Lĩnh vực'].dropna().unique()

@app.route('/')
def index():
    # Truyền danh sách lĩnh vực vào template HTML
    return render_template('index.html', fields=unique_fields)

@app.route('/submit', methods=['POST'])
def submit():
    family_advice = request.form.get('family_advice')
    financial_influence = request.form.get('financial_influence')
    family_industry = request.form.get('family_industry')

    # Xử lý kết quả và trả lại phản hồi cho người dùng
    return jsonify({
        'family_advice': family_advice,
        'financial_influence': financial_influence,
        'family_industry': family_industry
    })

if __name__ == '__main__':
    app.run(debug=True)
