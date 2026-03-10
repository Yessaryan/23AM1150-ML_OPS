from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pickle
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(
    request: Request,
    age: float = Form(...),
    sex: float = Form(...),
    cp: float = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: float = Form(...),
    restecg: float = Form(...),
    thalach: float = Form(...),
    exang: float = Form(...),
    oldpeak: float = Form(...),
):

    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])

    prediction = model.predict(features)

    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

    return {"prediction": result}