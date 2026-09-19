# Ganga AI – Ganga River Water Quality Forecasting System

## About

Ganga AI is a web-based application for monitoring and forecasting Ganga River water quality using machine learning.

The system analyzes water quality parameters such as:

- pH
- Temperature
- Dissolved Oxygen
- Turbidity
- BOD

## Features

- Water quality prediction using Machine Learning
- Random Forest and SVM model comparison
- IoT-based water quality monitoring
- Satellite data analysis
- Water quality assessment and recommendations
- PDF report generation
- Web-based dashboard

## Technologies Used

- Python
- Flask
- HTML & CSS
- Pandas
- Scikit-learn
- Joblib
- FPDF

## Project Files

- `App.py` – Main Flask application
- `water_quality.csv` – Water quality dataset
- `model.pkl` – Trained machine learning model
- `model_results.pkl` – Model performance results
- `satellite_data.py` – Satellite data module
- `report.pdf` – Project report
- `screenshots/` – Project screenshots

## Machine Learning

The project uses a trained Random Forest model to classify water quality into:

**Excellent | Healthy | Warning | Critical**

## Running the Project

Install the required packages:

```bash
pip install flask pandas scikit-learn joblib fpdf
