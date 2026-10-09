from flask import Flask, render_template, request
import joblib
import numpy as np

obj = joblib.load('california.joblib')

model = obj['model']
columns = obj['columns']

print("Columns:", columns)

app = Flask(__name__)


@app.route("/")
def Main():
    return render_template("index.html", columns=columns)


@app.route("/predict", methods=["POST"])
def predict():

    input_data = []

    for i in columns:
        value = request.form[i]
        input_data.append(float(value))

    prediction = model.predict([input_data])

    result = prediction[0]

    return render_template(
        "index.html",
        columns=columns,
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)