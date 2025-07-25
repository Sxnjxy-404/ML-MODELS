from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model_performance = pickle.load(open('model_performance.pkl', 'rb'))

# Performance label mapping
performance_map = {0: 'Poor', 1: 'Average', 2: 'Good'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance_percentage'])
        grade = float(request.form['previous_grade_numeric'])  # Assuming user enters 0 to 4

         # 2. Input validation
        if not (0 <= study_hours <= 12):
            return render_template("result.html", prediction="❌ Invalid Study Hours: Must be between 0 and 12.")
        if not (0 <= attendance <= 100):
            return render_template("result.html", prediction="❌ Invalid Attendance: Must be between 0 and 100.")
        if not (0 <= grade <= 4):
            return render_template("result.html", prediction="❌ Invalid GPA: Must be between 0.0 and 4.0.")
        
        # Format input
        input_data = [[study_hours, attendance, grade]]
        prediction = model_performance.predict(input_data)
        predicted_label = performance_map[prediction[0]]
        
        return render_template('result.html', prediction=predicted_label)
    
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
