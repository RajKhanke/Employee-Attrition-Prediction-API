# Employee Attrition Prediction API

This repository hosts an **Employee Attrition Prediction** model, developed using **FastAPI** and deployed on **Hugging Face**. The model predicts whether an employee is likely to stay or leave the company based on various input features. It uses a **K-Nearest Neighbors (KNN)** algorithm for prediction.

## Features

- **API Endpoint**: `/predict`
  - Takes employee information as input and returns a prediction about whether the employee will leave or stay with the company.
- **Machine Learning Model**: K-Nearest Neighbors (KNN)
- **Input Features**:
  - **Gender** (string)
  - **Age** (int)
  - **Department** (string)
  - **Job Title** (string)
  - **Years at Company** (int)
  - **Satisfaction Level** (float)
  - **Average Monthly Hours** (int)
  - **Promotion in Last 5 Years** (boolean)
  - **Salary** (int)

### Hosted API Link:
You can access the hosted API [here](https://rajkhanke-employee-attrition-api.hf.space/docs).

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/employee-attrition-prediction-api.git
cd employee-attrition-prediction-api
