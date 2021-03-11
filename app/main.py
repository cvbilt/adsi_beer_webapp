from fastapi import FastAPI, Query
from typing import List
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

#gmm_pipe 

@app.get("/")
def read_root():
    return 'Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo related to this project'


@app.get('/health', status_code=200)
def healthcheck():
    return 'Returning status code 200 with a string with a welcome message of your choice'

def format_features(genre: str,	age: int, income: int, spending: int):
  return {
        'Gender': [genre],
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending]
    }

@app.post("/beer/type/")
def predict(genre: str,	age: int, income: int, spending: int):
    features = format_features(genre,	age, income, spending)
    obs = pd.DataFrame(features)
    pred = gmm_pipe.predict(obs)
    # return JSONResponse(pred.tolist())
    return 'Returning prediction for a single input only & for a multiple inputs'


@app.get("/model/architecture/")
def arch():
    return 'Displaying the architecture of your Neural Networks (listing of all layers with their types)'


