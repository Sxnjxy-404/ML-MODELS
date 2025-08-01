from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# --- Load trained model and vectorizer ---
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# --- Home route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    user_input = ""
    if request.method == 'POST':
        user_input = request.form['message']
        message_vector = vectorizer.transform([user_input])
        result = model.predict(message_vector)[0]
        prediction = "Spam" if result == 1 else "Not Spam"

    return render_template('index.html', prediction=prediction, message=user_input)

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
