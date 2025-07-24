from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('movie_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        gender = request.form['gender']
        frequency = int(request.form['frequency'])
        genre = request.form['genre']

        # Remove emojis before mapping
        gender = gender.strip().split(' ')[-1]
        genre = genre.strip().split(' ')[-1]

        # Encode gender
        gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
        gender_encoded = gender_map.get(gender, 2)

        # Encode genre
        genre_map = {
            'Action': 0,
            'Comedy': 1,
            'Drama': 2,
            'Horror': 3,
            'Romance': 4,
            'Sci-Fi': 5,
            'Thriller': 6
        }
        genre_encoded = genre_map.get(genre, 0)

        # Prepare input for prediction
        input_data = np.array([[age, gender_encoded, frequency, genre_encoded]])
        prediction = model.predict(input_data)[0]

        # Output message
        result = "🎬 You are interested in movies!" if prediction == 1 else "😐 You seem less interested in movies."

        return render_template('result.html', result=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
