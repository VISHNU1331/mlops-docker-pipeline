from fastapi import FastAPI
import joblib
from pydantic import BaseModel

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

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: IrisInput):
    # Extract features from input
    features = [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]
    # Make prediction
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
