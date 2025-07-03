from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import zipfile
from io import BytesIO
from PIL import Image

app = FastAPI()

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
async def predict():
    # Dummy prediction response
    return {"prediction": "dummy_result"} 