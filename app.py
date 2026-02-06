from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# -------------------------------
# Dataset
# -------------------------------
data = {
    "age": [25, 30, 45, 60, 35, 50],
    "symptom": ["diabetes", "anemia", "hypertension", "diabetes", "obesity", "anemia"],
    "calories": [1800, 2200, 1600, 1500, 2000, 2100],
    "food": ["brown rice", "spinach", "oats", "millet", "salad", "dates"]
}

df = pd.DataFrame(data)

# Encoding
symptom_encoder = LabelEncoder()
food_encoder = LabelEncoder()

df["symptom"] = symptom_encoder.fit_transform(df["symptom"])
df["food"] = food_encoder.fit_transform(df["food"])

# Train model
X = df[["age", "symptom", "calories"]]
y = df["food"]

model = DecisionTreeClassifier()
model.fit(X, y)

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    food = None
    error = None

    if request.method == "POST":
        age = int(request.form["age"])
        symptom = request.form["symptom"].lower()
        calories = int(request.form["calories"])

        if symptom not in symptom_encoder.classes_:
            error = f"Invalid symptom. Allowed: {list(symptom_encoder.classes_)}"
        else:
            symptom_encoded = symptom_encoder.transform([symptom])[0]
            prediction = model.predict([[age, symptom_encoded, calories]])
            food = food_encoder.inverse_transform(prediction)[0]

    return render_template("index.html", food=food, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
