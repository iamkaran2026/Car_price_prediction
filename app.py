from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("car_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None

    if request.method == "POST":
        present_price = float(request.form["present_price"])
        kms_driven = int(request.form["kms_driven"])
        owner = int(request.form["owner"])
        year = int(request.form["year"])
        fuel_type = request.form["fuel_type"]
        seller_type = request.form["seller_type"]
        transmission = request.form["transmission"]

        # Calculate car age
        car_age = 2025 - year

        # Encode categories
        fuel_diesel = 1 if fuel_type == "Diesel" else 0
        fuel_petrol = 1 if fuel_type == "Petrol" else 0
        seller_individual = 1 if seller_type == "Individual" else 0
        transmission_manual = 1 if transmission == "Manual" else 0

        features = np.array([[present_price, kms_driven, owner, car_age,
                              fuel_diesel, fuel_petrol,
                              seller_individual, transmission_manual]])

        prediction = model.predict(features)
        predicted_price = round(prediction[0], 2)

    return render_template("index.html", predicted_price=predicted_price)


if __name__ == "__main__":
    app.run(debug=True)
