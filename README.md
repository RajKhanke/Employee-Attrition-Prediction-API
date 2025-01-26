# Employee Attrition Prediction API

![Employee Attrition](https://img.shields.io/badge/Employee%20Attrition-Prediction-brightgreen)

This repository hosts an **Employee Attrition Prediction** model, developed using **FastAPI** and deployed on **Hugging Face**. The model predicts whether an employee is likely to stay or leave the company based on various input features. It uses a **K-Nearest Neighbors (KNN)** algorithm for prediction.

## 🚀 Features

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

### 🌐 Hosted API Link:
You can access the hosted API [here](https://rajkhanke-employee-attrition-api.hf.space/docs).

---

## 🛠 Setup and Installation

### 1. Clone the repository

To set up this project locally, you need to clone the repository to your local machine.

```bash
git clone https://github.com/yourusername/employee-attrition-prediction-api.git
cd employee-attrition-prediction-api
```
### 2. Install required dependencies

Before running the application, install the necessary dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI server locally

To run the  `FastAPI ` server locally, you can use  `Uvicorn`, which is a high-performance ASGI server.
```bash
uvicorn app:main --reload --host 0.0.0.0 --port 8000

```

The API server will be available at http://127.0.0.1:8000.

## 🌟 API Routes

### 1. `/predict` (POST)
This endpoint predicts whether an employee will leave the company based on their profile information.

### Request Body Example:
```json
{
  "Gender": "Female",
  "Age": 30,
  "Department": "Sales",
  "Job_Title": "Analyst",
  "Years_at_Company": 3,
  "Satisfaction_Level": 80.5,
  "Average_Monthly_Hours": 160,
  "Promotion_Last_5Years": "No",
  "Salary": 50000
}
```
### Response Example:
```json
{
  "prediction": "The employee is likely to stay in the company."
}
```

### 4. `/docs` (GET)


**🧪 Using Swagger UI (FASTAPI) :**

The Swagger UI documentation automatically generated by FastAPI. Use this to explore and test all available API endpoints interactively./docs (GET)

To test the  `API ` server locally using swaggerUI,  you can use  `Run`, the following Url on Server :
```bash
http://127.0.0.1:8000/docs
```
![image](https://github.com/RajKhanke/employee_attrition_prediction_API_FASTAPI/blob/main/Screenshot%202025-01-26%20133422.png)
![image](https://github.com/RajKhanke/employee_attrition_prediction_API_FASTAPI/blob/main/Screenshot%202025-01-26%20133422.png)



**🧪 Using Postman or CURL:**

- Example `CURL ` Request:

```bash
curl -X 'POST' \
'http://127.0.0.1:8000/predict' \
-H 'Content-Type: application/json' \
-d '{
  "Gender": "Female",
  "Age": 30,
  "Department": "Sales",
  "Job_Title": "Analyst",
  "Years_at_Company": 3,
  "Satisfaction_Level": 80.5,
  "Average_Monthly_Hours": 160,
  "Promotion_Last_5Years": "No",
  "Salary": 50000
}'
```

- Expected Response:


```bash
{
  "prediction": "The employee is likely to stay in the company."
}
```

### 💡 REST API Fundamentals



**What is FastAPI?**

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python. It is based on standard Python type hints and provides:

- Automatic validation

- Automatic documentation (Swagger UI)

- Interactive API testing




**Key Features of FastAPI:**

- `Fast`: Quick to code, fast to run, and easy to get started.

- `Automatic Documentation`: Swagger UI and ReDoc are auto-generated.

- `Validation`: Pydantic models validate input data and provide detailed error messages.

- `Asynchronous Support`: Allows sync and async endpoints for flexible request handling.




**Request Methods:**

-  `POST:` Used to send data to the server (e.g., /predict).

- `GET:` Used to retrieve data from the server (e.g., /docs).




### 🌍 Deploying the API on Hugging Face

**Steps to Deploy:**

- Create a Hugging Face Account:

- Sign up at Hugging Face.

- Link your Hugging Face account with GitHub to host the project.

- Create a Space:

- Navigate to the `Spaces ` section of your Hugging Face profile.

- Create a new space and choose the `Docker` template.

- Push Your Code to Hugging Face:

- Initialize a Git repository in the project folder (if not already done).

- Push the code to Hugging Face using the following commands:

```bash
git init
git remote add origin https://huggingface.co/spaces/yourusername/yourrepo
git add .
git commit -m "Initial commit"
git push origin main
```


**Verify Deployment:**

- Hugging Face will automatically build and deploy your API.

- Visit the Space's URL to access the hosted API.




### 🎉 Contribution

Feel free to fork the repository and submit pull requests. When contributing, please adhere to these guidelines:

-  `Code Style `: Follow the PEP 8 style guide.

-  `Testing `: Ensure all new features are covered with tests.

-  `Documentation `: Update the documentation if you add new features or make changes.




### 📞 Contact

- For any questions or inquiries, feel free to reach out via open an issue in the repository.



### 💖 Acknowledgments

 `FastAPI `: For providing an easy-to-use framework for building APIs.

 `Hugging Face `: For enabling simple deployment of machine learning models.

 `Scikit-learn `: For providing machine learning algorithms like KNN.

 `Uvicorn `: For serving the FastAPI application with high performance.

