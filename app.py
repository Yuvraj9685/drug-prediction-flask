from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# -----------------------------
# Load trained ML components
# -----------------------------
model = joblib.load("drug_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("label_encoder.joblib")

# -----------------------------
# UI ROUTE (HTML FORM)
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            age = int(request.form["age"])
            sex = int(request.form["sex"])
            bp = int(request.form["bp"])
            cholesterol = int(request.form["cholesterol"])
            na_to_k = float(request.form["na_to_k"])

            input_data = np.array([[age, sex, bp, cholesterol, na_to_k]])
            input_scaled = scaler.transform(input_data)

            pred = model.predict(input_scaled)
            prediction = encoder.inverse_transform(pred)[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

# -----------------------------
# REST API ROUTE (JSON)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        age = int(data["age"])
        sex = int(data["sex"])
        bp = int(data["bp"])
        cholesterol = int(data["cholesterol"])
        na_to_k = float(data["na_to_k"])

        input_data = np.array([[age, sex, bp, cholesterol, na_to_k]])
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)
        drug = encoder.inverse_transform(pred)[0]

        return jsonify({
            "prediction": drug,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        })

# -----------------------------
# APP ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

