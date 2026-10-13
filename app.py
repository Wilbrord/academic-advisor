from math import isfinite
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "performance_risk_model.pkl"
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
features = model_bundle["features"]
default_values = {
  "gpa": "2.80",
  "attendance_pct": "85",
  "weekly_study_hours": "12",
  "credits_enrolled": "15",
  "prior_failed_courses": "0",
}


def make_suggestions(values: dict[str, float]) -> list[str]:
  suggestions = []
  if values["gpa"] < 2.5:
    suggestions.append("Book time with an academic advisor to review course progress.")
  if values["attendance_pct"] < 75:
    suggestions.append("Meet course instructors and make a plan for regular attendance.")
  if values["weekly_study_hours"] < 10:
    suggestions.append("Set a weekly study schedule and use tutoring or peer study sessions.")
  if values["credits_enrolled"] > 18:
    suggestions.append("Review your course load with an advisor before the add/drop deadline.")
  if values["prior_failed_courses"] >= 2:
    suggestions.append("Check repeat-course rules and arrange support for previously difficult subjects.")
  if not suggestions:
    suggestions.append("Keep monitoring your progress and use instructor office hours when useful.")
  return suggestions


@app.route("/", methods=["GET", "POST"])
def home():
  values = default_values.copy()
  result = None
  error = None

  if request.method == "POST":
    values = {name: request.form.get(name, "").strip() for name in default_values}
    try:
      numeric_values = {name: float(value) for name, value in values.items()}
      if not all(isfinite(value) for value in numeric_values.values()):
        raise ValueError
      if not 0 <= numeric_values["gpa"] <= 4:
        raise ValueError
      if not 0 <= numeric_values["attendance_pct"] <= 100:
        raise ValueError
      if not 0 <= numeric_values["weekly_study_hours"] <= 40:
        raise ValueError
      if not 6 <= numeric_values["credits_enrolled"] <= 21:
        raise ValueError
      if not 0 <= numeric_values["prior_failed_courses"] <= 6:
        raise ValueError
      if any(
        numeric_values[name] % 1 != 0
        for name in ("credits_enrolled", "prior_failed_courses")
      ):
        raise ValueError

      student = pd.DataFrame([numeric_values], columns=features)
      risk_probability = float(model.predict_proba(student)[0, 1])
      if risk_probability >= 0.65:
        risk_level = "High"
      elif risk_probability >= 0.35:
        risk_level = "Watch"
      else:
        risk_level = "Lower"
      result = {
        "probability": round(risk_probability * 100),
        "level": risk_level,
        "suggestions": make_suggestions(numeric_values),
      }
    except ValueError:
      error = "Enter valid values within the ranges shown. Credits and failed courses must be whole numbers."

  return render_template("index.html", values=values, result=result, error=error)


@app.get("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(debug=True)