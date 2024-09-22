from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("models/iris_model.pkl")

# Define the input data model
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define class names for the Iris dataset
class_names = ['Setosa', 'Versicolor', 'Virginica']

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: IrisInput):
    try:
        # Extract features from input
        features = [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]
        
        # Check for valid input dimensions
        if len(features[0]) != 4:
            raise ValueError("Input data must have four features.")
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return descriptive class name
        return {"prediction": class_names[int(prediction[0])]}
    
    except Exception as e:
        # Raise HTTP 400 for bad requests with a detailed error message
        raise HTTPException(status_code=400, detail=str(e))
