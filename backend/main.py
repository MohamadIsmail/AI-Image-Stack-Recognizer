from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import zipfile
from io import BytesIO
from PIL import Image
from model import load_model, predict_image, load_openvino_model, predict_image_openvino

app = FastAPI()

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load models at startup
load_model()
load_openvino_model()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        return JSONResponse(status_code=400, content={"error": "Only .zip files are allowed."})

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """uses PyTorch for CPU inference"""
    if not (file.filename.lower().endswith('.png') or file.filename.lower().endswith('.jpg') or file.filename.lower().endswith('.jpeg')):
        return JSONResponse(status_code=400, content={"error": "Only image files (.png, .jpg, .jpeg) are allowed."})
    
    contents = await file.read()
    result = predict_image(contents)
    return result

@app.post("/predict_opt")
async def predict_optimized(file: UploadFile = File(...)):
    """uses OpenVINO for optimized CPU inference """
    if not (file.filename.lower().endswith('.png') or file.filename.lower().endswith('.jpg') or file.filename.lower().endswith('.jpeg')):
        return JSONResponse(status_code=400, content={"error": "Only image files (.png, .jpg, .jpeg) are allowed."})
    
    contents = await file.read()
    result = predict_image_openvino(contents)
    return result 