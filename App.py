from flask import Flask, render_template, request, redirect, send_file
from flask import session
import joblib
import random
from satellite_data import get_satellite_data
from fpdf import FPDF
import os


app = Flask(__name__, template_folder="templates")
app.secret_key = "ganga_water_project"

# Load ML model
model = joblib.load("model.pkl")
results = joblib.load("model_results.pkl")


# Temporary user database
users = {}
latest_report = {}


# ---------------- IoT Sensor Data Simulation ----------------

def get_iot_data():

    data = {
        "pH": round(random.uniform(6.5, 8.5), 2),
        "temperature": round(random.uniform(20, 35), 2),
        "dissolved_oxygen": round(random.uniform(5, 10), 2),
        "turbidity": round(random.uniform(1, 10), 2),
        "BOD": round(random.uniform(1, 5), 2)
    }

    return data



# ---------------- Home ----------------

@app.route("/")
def home():
    return render_template("login.html")


@app.route("/home")
def homepage():
    return render_template("index.html")
# ---------------- Register ----------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]


        users[username] = {

            "email": email,
            "password": password

        }


        return render_template(
            "login.html",
            message="Registration successful. Please login."
        )


    return render_template("register.html")




# ---------------- Login ----------------

@app.route("/login", methods=["POST"])
def login():

    username = request.form["username"]
    password = request.form["password"]

    if username in users and users[username]["password"] == password:

        return redirect("/home")

    else:

        return render_template(
            "login.html",
            error="Invalid Username or Password"
        )


# ---------------- Manual Prediction ----------------

@app.route("/predict")
def predict():

    import pandas as pd
    import random

    # Read dataset
    df = pd.read_csv("dataset/water_quality.csv")

    # Select one random record
    sample = df.sample(1)
    date = sample.iloc[0]["Date"]
    station = sample.iloc[0]["Station"]

    # Remove target column
    X = sample[[
    "pH",
    "Temperature",
    "Dissolved_Oxygen",
    "Turbidity",
    "BOD"
  ]]
    # Predict
    prediction = model.predict(X)[0]

    # Read parameter values
    pH = sample.iloc[0]["pH"]
    temperature = sample.iloc[0]["Temperature"]
    dissolved_oxygen = sample.iloc[0]["Dissolved_Oxygen"]
    turbidity = sample.iloc[0]["Turbidity"]
    BOD = sample.iloc[0]["BOD"]

    # River Health Score
    if prediction == "Excellent":
        score = random.randint(90, 100)
        risk = "Low"
        forecast = "Excellent"

        recommendation = [
            "Continue weekly monitoring.",
            "Water quality is excellent.",
            "Safe for aquatic life.",
            "Maintain current pollution control measures."
        ]

    elif prediction == "Healthy":
        score = random.randint(75, 89)
        risk = "Medium"
        forecast = "Healthy"

        recommendation = [
            "Increase monitoring frequency.",
            "Inspect nearby pollution sources.",
            "Monitor dissolved oxygen regularly.",
            "Collect additional water samples."
        ]

    elif prediction == "Warning":
        score = random.randint(50, 74)
        risk = "High"
        forecast = "Warning"

        recommendation = [
            "Investigate possible pollution sources.",
            "Reduce untreated wastewater discharge.",
            "Increase sampling frequency.",
            "Take preventive environmental measures."
        ]

    else:
        score = random.randint(20, 49)
        risk = "Critical"
        forecast = "Critical"

        recommendation = [
            "Immediate inspection required.",
            "Notify Pollution Control Board.",
            "Stop untreated industrial discharge.",
            "Start emergency water quality assessment."
        ]


    global latest_report

    latest_report = {
        "date": date,
        "station": station,
        "prediction": prediction,
        "score": score,
        "risk": risk,
        "forecast": forecast,
        "pH": pH,
        "temperature": temperature,
        "dissolved_oxygen": dissolved_oxygen,
        "turbidity": turbidity,
        "BOD": BOD,
        "recommendation": recommendation
    }
    return render_template(

        "result.html",
        date=date,
        station=station,
        prediction=prediction,

        score=score,

        risk=risk,

        forecast=forecast,

        recommendation=recommendation,

        pH=pH,

        temperature=temperature,

        dissolved_oxygen=dissolved_oxygen,

        turbidity=turbidity,

        BOD=BOD

    )
# ---------------- IoT + Satellite Prediction ----------------

@app.route("/iot_predict", methods=["POST"])
def iot_predict():


    sensor = get_iot_data()

    satellite = get_satellite_data()



    prediction = model.predict([[

        sensor["pH"],
        sensor["temperature"],
        sensor["dissolved_oxygen"],
        sensor["turbidity"],
        sensor["BOD"]

    ]])


    result = prediction[0]



    if result == 0:

        result = "Poor Water Quality"

    elif result == 1:

        result = "Moderate Water Quality"

    else:

        result = "Good Water Quality"




    if result == "Poor Water Quality":

        score = 40
        recommendation = "High pollution detected. Treatment required before use."


    elif result == "Moderate Water Quality":

        score = 70
        recommendation = "Water quality is moderate. Continuous monitoring is recommended."


    else:

        score = 90
        recommendation = "Water quality is good. Suitable for normal use."




    return render_template(

        "index.html",

        prediction=result,
        score=score,
        values=list(sensor.values()),
        labels=list(sensor.keys()),
        satellite=satellite,
        recommendation=recommendation

    )




# ---------------- Pages ----------------
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login_page")

@app.route("/input")
def input_page():

    return render_template("prediction.html")



@app.route("/monitoring")
def monitoring():

    sensor = get_iot_data()

    return render_template(

        "monitoring.html",
        sensor=sensor

    )

# ---------------- Home Feature Pages ----------------


@app.route("/prediction")
def prediction():

    return render_template("prediction.html")



@app.route("/iot")
def iot():

    return render_template(
        "iot.html",
        report=latest_report
    )



@app.route("/analysis")
def analysis():

    return render_template(
        "analysis.html",
        report=latest_report
    )



@app.route("/report")
def report():

    return render_template("report.html")

@app.route("/satellite")
def satellite():

    return render_template(
        "satellite.html",
        report=latest_report
    )

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        report=latest_report
    )



# ---------------- Model Performance ----------------


@app.route("/performance")
def performance():


    rf_accuracy = round(results["Random Forest Accuracy"] * 100, 2)

    svm_accuracy = round(results["SVM Accuracy"] * 100, 2)



    if rf_accuracy >= svm_accuracy:

        best_model = "Random Forest"

    else:

        best_model = "SVM"



    return render_template(

        "performance.html",

        rf_accuracy=rf_accuracy,
        svm_accuracy=svm_accuracy,
        best_model=best_model

    )

@app.route("/download_report")
def download_report():

    global latest_report

    if not latest_report:
        return "Please generate a prediction first."

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, "Ganga AI Water Quality Prediction Report", ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    pdf.cell(190, 10, f"Date: {latest_report['date']}", ln=True)
    pdf.cell(190, 10, f"Station: {latest_report['station']}", ln=True)
    pdf.cell(190, 10, f"Prediction: {latest_report['prediction']}", ln=True)
    pdf.cell(190, 10, f"River Health Score: {latest_report['score']}", ln=True)
    pdf.cell(190, 10, f"Risk Level: {latest_report['risk']}", ln=True)
    pdf.cell(190, 10, f"Forecast: {latest_report['forecast']}", ln=True)

    pdf.ln(5)

    pdf.cell(190, 10, f"pH: {latest_report['pH']}", ln=True)
    pdf.cell(190, 10, f"Temperature: {latest_report['temperature']}", ln=True)
    pdf.cell(190, 10, f"Dissolved Oxygen: {latest_report['dissolved_oxygen']}", ln=True)
    pdf.cell(190, 10, f"Turbidity: {latest_report['turbidity']}", ln=True)
    pdf.cell(190, 10, f"BOD: {latest_report['BOD']}", ln=True)

    pdf.ln(5)
    pdf.cell(190, 10, "Recommendations:", ln=True)

    for item in latest_report["recommendation"]:
        pdf.multi_cell(190, 8, "- " + item)

    filename = "report.pdf"
    pdf.output(filename)

    return send_file(filename, as_attachment=True)

# ---------------- Run ----------------


if __name__ == "__main__":

    app.run(debug=True)