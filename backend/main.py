from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # For now, just return the filename
    return {"filename": file.filename}

@app.post("/predict")
async def predict():
    # Dummy prediction response
    return {"prediction": "dummy_result"} 