# 🚀 Mini Production ML System for Customer Churn Prediction

> A production-inspired Machine Learning project demonstrating an end-to-end ML lifecycle for Customer Churn Prediction, including data ingestion, feature engineering, model training, model serving with FastAPI, monitoring, drift detection, retraining strategy, and deployment-ready project structure.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-success)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 📌 Project Overview

This project demonstrates how a **Machine Learning model can be designed and deployed as a mini production system** rather than just being trained inside a notebook.

The objective is to predict whether a telecom customer is likely to **churn** using customer demographic, subscription, billing, and service usage information.

Unlike a traditional ML notebook, this project simulates an industry-standard Machine Learning workflow by implementing:

- Automated data ingestion
- Feature engineering
- Reproducible training pipeline
- Baseline vs Candidate model comparison
- Model artifact management
- REST API serving using FastAPI
- Performance benchmarking
- Data quality monitoring
- Feature drift detection
- Retraining strategy
- Production architecture

This project follows modern **Machine Learning Engineering** and **MLOps** best practices.

---

# 🎯 Problem Statement

Customer churn is one of the biggest challenges for telecom companies.

The goal is to predict whether a customer is likely to leave the company so that proactive retention strategies can be applied.

This project formulates the problem as a **Binary Classification** task.

Target Variable:

```
Churn
```

Prediction Classes

- Yes → Customer will churn
- No → Customer will stay

---

# 📊 Dataset

Dataset Used:

**Telco Customer Churn Dataset**

The dataset contains customer information including

- Customer demographics
- Contract details
- Internet services
- Payment information
- Monthly charges
- Total charges
- Tenure
- Additional service subscriptions

---

# ⚙️ Technology Stack

### Programming

- Python

### Machine Learning

- Scikit-Learn
- Joblib

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### API

- FastAPI
- Uvicorn

### Configuration

- YAML

### Testing

- PyTest

---

# 🏗 Production ML Pipeline

```
Raw Customer Data
        │
        ▼
Batch Data Ingestion
        │
        ▼
Data Validation
        │
        ▼
Feature Engineering
        │
        ▼
Train / Validation Split
        │
        ▼
Baseline Model
        │
        ▼
Candidate Model
        │
        ▼
Model Evaluation
        │
        ▼
Save Model Artifact
        │
        ▼
FastAPI Inference Service
        │
        ▼
Monitoring
        │
        ▼
Drift Detection
        │
        ▼
Retraining Trigger
```

---

# ✨ Project Features

## ✅ Data Engineering

- Batch data ingestion
- Data validation
- Missing value handling
- Shared preprocessing pipeline
- Feature engineering

---

## ✅ Machine Learning

- Baseline Model
- Candidate Model
- Model comparison
- ROC-AUC evaluation
- Classification metrics
- Confusion Matrix
- Model persistence

---

## ✅ Model Serving

REST API built using FastAPI

Endpoints

```
GET /

GET /health

POST /predict
```

Example Prediction Request

```json
{
  "gender":"Female",
  "SeniorCitizen":0,
  "Partner":"Yes",
  "Dependents":"No",
  "tenure":18,
  "PhoneService":"Yes",
  "InternetService":"Fiber optic",
  "MonthlyCharges":88.4,
  "TotalCharges":1540.2
}
```

---

# 📈 Model Evaluation

Metrics Used

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Model Selection Strategy

The project compares

- Baseline Model
- Candidate Model

The candidate model is promoted only when it satisfies the predefined performance threshold.

---

# ⚡ Performance Benchmark

The API performance benchmark reports

- Average Latency
- P95 Latency
- Throughput
- Total Requests Processed

These measurements simulate production inference performance.

---

# 📊 Monitoring Strategy

The project includes a lightweight monitoring framework.

It tracks

### Infrastructure Metrics

- API latency
- Error rate
- Throughput

### Data Quality

- Missing values
- Invalid values
- Schema validation

### Feature Drift

- Mean comparison
- Standard deviation comparison

### Model Monitoring

- Prediction distribution
- Accuracy (when labels become available)

---

# 🔄 Retraining Strategy

The model is retrained whenever one or more of the following conditions are met.

- Significant feature drift
- Accuracy degradation
- Large amount of new customer data
- Scheduled retraining interval

---

# 🧪 Project Structure

```
mini-production-ml-system-telco-churn
│
├── notebooks
│   └── Mini_Production_ML_System.ipynb
│
├── data
│   ├── raw
│   ├── processed
│   └── recent_batch
│
├── src
│   ├── feature_engineering.py
│   ├── train.py
│   ├── inference.py
│   ├── monitor.py
│   ├── ingest.py
│   └── utils.py
│
├── api
│   └── app.py
│
├── models
│   ├── churn_model.pkl
│   └── model_metadata.json
│
├── artifacts
│   ├── evaluation_report.json
│   ├── metrics.csv
│   └── confusion_matrix.png
│
├── tests
│   └── test_pipeline.py
│
├── configs
│   └── config.yaml
│
├── reports
│   ├── Mini_Production_ML_System_Design_Report.docx
│   └── Architecture_Diagram.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ▶️ Running the Project

## Clone Repository

```bash
git clone https://github.com/yourusername/mini-production-ml-system-telco-churn.git
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python src/train.py
```

---

## Run FastAPI Server

```bash
uvicorn api.app:app --reload
```

---

## Open API Documentation

```
http://127.0.0.1:8000/docs
```

---

# 📷 Demo

The repository contains:

- Training Pipeline
- API Prediction
- Monitoring Output
- Drift Detection
- Performance Benchmark
- Architecture Diagram

---

# 🎓 Learning Outcomes

This project demonstrates practical implementation of:

- Production Machine Learning
- Feature Engineering
- Machine Learning Pipelines
- Model Evaluation
- Model Versioning
- REST API Deployment
- Data Quality Monitoring
- Feature Drift Detection
- Retraining Strategy
- Machine Learning System Design
- Reproducible ML Workflows

---

# 🌟 Future Improvements

- Docker containerization
- CI/CD using GitHub Actions
- MLflow Model Registry
- Kubernetes deployment
- Prometheus + Grafana monitoring
- Feature Store integration
- Cloud deployment (AWS/Azure/GCP)
- Automated retraining pipeline
- Canary model deployment

---

# 👨‍💻 Author

**Akash Ghosh**

Passionate about Machine Learning Engineering, Data Science, MLOps, and Production AI Systems.

If you found this project useful, consider giving it a ⭐ on GitHub!

---
