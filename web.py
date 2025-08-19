from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import joblib

# ✅ Home route

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Collect input values
        input_features = ["NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
        input_data = [float(request.form[feature]) for feature in input_features]

        # Load model and scaler
        model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Scale and predict
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]

        # Define threshold
        threshold = 5.0

        # Render result page
        return render_template('result.html', prediction=prediction, threshold=threshold)

    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
