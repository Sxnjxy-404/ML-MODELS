from flask import Flask, render_template, request, redirect, send_file, flash
from werkzeug.utils import secure_filename
from invoice_processor import process_invoice
import os
import csv
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'invoice' not in request.files:
        flash('No file part')
        return redirect('/')

    file = request.files['invoice']
    if file.filename == '':
        flash('No selected file')
        return redirect('/')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        extracted_data = process_invoice(filepath)

        # ✅ Save CSV in memory and write to uploads folder
        csv_file = io.StringIO()
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Field', 'Value'])
        for key, value in extracted_data.items():
            csv_writer.writerow([key, value])
        csv_file.seek(0)

        with open('uploads/invoice_data.csv', 'w', newline='', encoding='utf-8') as f:
            f.write(csv_file.getvalue())

        return render_template('result.html', data=extracted_data, filename=filename)

    else:
        flash('Invalid file type')
        return redirect('/')

@app.route('/download_csv')
def download_csv():
    try:
        return send_file('uploads/invoice_data.csv', as_attachment=True)
    except Exception as e:
        return f"Error downloading file: {e}"

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
