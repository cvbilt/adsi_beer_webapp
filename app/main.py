from fastapi import FastAPI, Query
from typing import List
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

#models = load('../models/.joblib')

@app.get("/")
def read_root():
    return 'Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo related to this project'


@app.get('/health', status_code=200)
async def healthcheck():
    return 'The Beer app is ready to run'

def format_features(b_name: str, r_aroma: int, r_appearance: int, r_palate: int, r_taste: int):
  return {
        'Brewery Name': [b_name],
        'Review Aroma': [r_aroma],
        'Review Appearance': [r_appearance],
        'Review Palate': [r_palate],
        'Review Taste': [r_taste]
    }

@app.post("/beer/type/")
def predict(b_name: str, r_aroma: int, r_appearance: int, r_palate: int, r_taste: int):
    features = format_features(b_name, r_aroma, r_appearance, r_palate, r_taste)
    obs = pd.DataFrame(features)
    pred = models.predict(obs)
    return JSONResponse(pred.tolist())


@app.get("/model/architecture/")
def arch():
    return 'Displaying the architecture of your Neural Networks (listing of all layers with their types)'


