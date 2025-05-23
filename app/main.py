from fastapi import FastAPI, UploadFile, File
from app.model import predict_face_problem

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Face Problem Analyzer API is live."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = await predict_face_problem(file)
    return result
