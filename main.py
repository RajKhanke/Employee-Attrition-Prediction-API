from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model
model_knn = joblib.load("knn_model.pkl")

# Load label encoders
gender_encoder = LabelEncoder()
promotion_encoder = LabelEncoder()

gender_encoder.fit(['Male', 'Female'])
promotion_encoder.fit(['Yes', 'No'])

# Define the input data model
class EmployeeData(BaseModel):
    Gender: str
    Age: int
    Department: str
    Job_Title: str
    Years_at_Company: int
    Satisfaction_Level: int
    Average_Monthly_Hours: int
    Promotion_Last_5Years: str
    Salary: int

# Preprocess input function
def preprocess_input(data: pd.DataFrame):
    # Encode categorical variables
    data['Gender'] = gender_encoder.transform(data['Gender'])
    data['Promotion_Last_5Years'] = promotion_encoder.transform(data['Promotion_Last_5Years'])

    # Create dummy columns for Department and Job_Title
    data = pd.get_dummies(data, columns=['Department', 'Job_Title'], drop_first=True)

    # Ensure the columns are in the same order as the training data
    X_columns = ['Age', 'Gender', 'Years_at_Company', 'Satisfaction_Level', 'Average_Monthly_Hours',
                 'Promotion_Last_5Years', 'Salary', 'Department_Finance', 'Department_HR',
                 'Department_Marketing', 'Department_Sales', 'Job_Title_Analyst',
                 'Job_Title_Engineer', 'Job_Title_HR Specialist', 'Job_Title_Manager']

    data = data.reindex(columns=X_columns, fill_value=0)
    return data

@app.post("/predict")
def predict_attrition(employee: EmployeeData):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([employee.dict()])

    # Preprocess the input data
    processed_data = preprocess_input(input_data)

    # Make prediction
    prediction = model_knn.predict(processed_data)

    # Interpret prediction
    result = "The employee is likely to leave the company." if prediction[0] == 1 else "The employee is likely to stay in the company."

    return {"prediction": result}
