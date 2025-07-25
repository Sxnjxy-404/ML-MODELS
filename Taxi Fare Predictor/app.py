from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("taxi_model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        population = float(request.form["population"])
        monthly_income = float(request.form["monthly_income"])
        parking_cost = float(request.form["parking"])
        riders = float(request.form["riders"])

        # Prepare input for prediction
        features = np.array([[population, monthly_income, parking_cost, riders]])
        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
