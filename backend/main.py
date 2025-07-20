from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import zipfile
from io import BytesIO
from PIL import Image
from model import load_model, predict_image, load_openvino_model, predict_batch_openvino, predict_batch_pytorch
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files directory
app.mount("/uploaded_images", StaticFiles(directory=UPLOAD_DIR), name="uploaded_images")

# Load models at startup
load_model()
load_openvino_model()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        return JSONResponse(status_code=400, content={"error": "Only .zip files are allowed."})

    # Clean the uploaded_images folder before extracting new images
    for f in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Read the uploaded zip file into memory
    contents = await file.read()
    zip_bytes = BytesIO(contents)
    with zipfile.ZipFile(zip_bytes) as zip_ref:
        image_filenames = [name for name in zip_ref.namelist() if name.lower().endswith('.png')]
        resolutions = []
        extracted_files = []
        for name in image_filenames:
            with zip_ref.open(name) as img_file:
                img = Image.open(img_file)
                resolution = img.size  # (width, height)
                resolutions.append(resolution)
                # Save the image to UPLOAD_DIR
                save_path = os.path.join(UPLOAD_DIR, os.path.basename(name))
                img.save(save_path)
                extracted_files.append(os.path.basename(name))

    metadata = {
        "num_images": len(extracted_files),
        "filenames": extracted_files,
        "resolutions": resolutions
    }
    return metadata

class BatchPredictionRequest(BaseModel):
    filenames: List[str]
    batch_size: Optional[int] = 8

@app.post("/predict")
async def predict(request: BatchPredictionRequest):
    """uses PyTorch for CPU inference (single or batch)"""
    if not request.filenames:
        return JSONResponse(status_code=400, content={"error": "No filenames provided."})

    # Validate that all files exist
    missing_files = []
    for filename in request.filenames:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)

    if missing_files:
        return JSONResponse(
            status_code=400,
            content={"error": f"Files not found: {', '.join(missing_files)}"}
        )

    # Run batch prediction (even for single file)
    batch_size = request.batch_size if hasattr(request, 'batch_size') else None
    results = predict_batch_pytorch(request.filenames, batch_size=batch_size)
    return results

@app.post("/predict_optimized")
async def predict_optimized_batch(request: BatchPredictionRequest):
    """uses OpenVINO for optimized CPU inference on multiple images"""
    if not request.filenames:
        return JSONResponse(status_code=400, content={"error": "No filenames provided."})
    
    # Validate that all files exist
    missing_files = []
    for filename in request.filenames:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    if missing_files:
        return JSONResponse(
            status_code=400, 
            content={"error": f"Files not found: {', '.join(missing_files)}"}
        )
    
    # Run batch prediction with configurable batch_size
    batch_size = request.batch_size if request.batch_size else 8
    results = predict_batch_openvino(request.filenames, batch_size=batch_size)
    return results 