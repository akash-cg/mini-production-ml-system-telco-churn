# 🚀 Mini Production ML System for Customer Churn Prediction

> A production-inspired Machine Learning project demonstrating an end-to-end ML lifecycle for Customer Churn Prediction, including data ingestion, feature engineering, model training, model serving with FastAPI, monitoring, drift detection, retraining strategy, and deployment-ready project structure.


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
---

