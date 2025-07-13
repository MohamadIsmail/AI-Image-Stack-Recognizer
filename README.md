# AI-Image-Stack-Recognizer
A full-stack web app for uploading and viewing image stacks interactively. Also the user can run AI inference on selected images using a deep learning image recognition model optimized for CPU inference.


## Features

- **File Upload**: Select and upload ZIP files containing PNG images
- **Interactive Viewer**: Browse through uploaded images with navigation controls
- **Image Information**: Display filename, resolution, and position in stack
- **Thumbnail Navigation**: Quick access to any image in the stack
- **Responsive Design**: Works on desktop and mobile devices
- **AI Inference**: Run AI model inference (ResNet18) on selected images, with support for both single-image and batch (optimized) predictions


## Tech Stack

- **Docker** – Containerization
### Frontend
- **React 18** – UI framework
- **Tailwind CSS** – Styling
- **Axios** – HTTP client
- **Nginx** – Static file server (in production/Docker)

### Backend
- **FastAPI** – Python web framework for API
- **Uvicorn** – ASGI server for FastAPI
- **Pillow** – Image processing
- **PyTorch** – Deep learning inference (ResNet18 model)
- **Torchvision** – Model utilities and transforms
- **OpenVINO** – Optimized inference for batch predictions on CPU
- **Python 3.12** – Runtime



## Setup

### Backend

#### Bare Machine

1. **Install Python 3.12** (or compatible version).
2. **Install dependencies:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Run the backend:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

#### Docker

1. **Build and run the backend container:**
   ```bash
   cd backend
   docker build -t ai-image-stack-backend .
   docker run -p 8000:8000 ai-image-stack-backend
   ```

---

### Frontend

#### Bare Machine

1. **Install Node.js (v18 recommended) and npm.**
2. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```
3. **Start the development server:**
   ```bash
   npm start
   ```
   The app will be available at [http://localhost:3000](http://localhost:3000).

4. **Build for production:**
   ```bash
   npm run build
   serve -s build
   ```

#### Docker

1. **Build and run the frontend container:**
   ```bash
   cd frontend
   docker build -t ai-image-stack-frontend .
   docker run -p 3000:80 ai-image-stack-frontend
   ```
   The app will be available at [http://localhost:3000](http://localhost:3000).

---

### Using Docker Compose

To run both backend and frontend together:
```bash
docker-compose up --build
```
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend:  [http://localhost:8000](http://localhost:8000)

## Usage

1. **Upload Images**:
   - Click "Select ZIP File" to choose a ZIP file containing PNG images
   - Click "Upload Images" to upload the file to the backend
   - Wait for the upload to complete

2. **View Images**:
   - After successful upload, the Image Stack Viewer will appear
   - Use the arrow buttons or arrow keys to navigate through images
   - Click on thumbnails to jump to specific images
   - View image information including filename, resolution, and position
   - **Zoom and Rotate**: Use the zoom slider and rotate buttons to inspect images in detail and adjust their orientation interactively

3. **Inference on Selected Images**:
   - Select one or more images using the checkboxes in the viewer or thumbnail strip
   - Click the "Run Inference on Selected Images" button to send the selected images to the backend for Optimized AI model prediction
   - View the inference results (class and confidence) displayed alongside each image
   - Batch inference is supported for multiple images, and single-image inference is also available.

## API Endpoints

The frontend communicates with the backend API:

- `POST /upload` - Upload ZIP file containing images
- `/uploaded_images/{filename}` - Serves static image files that have been uploaded to the backend.

**/predict**
- Endpoint: `POST /predict`
- Description: Run AI inference on one or more images using the standard PyTorch ResNet18 model (CPU).
- **This endpoint is provided for experimentation and to compare results and performance before and after model optimization. It is not used by the web app.**
- Request body (JSON):
  ```json
  {
    "filenames": ["image1.png", "image2.png"],
    "batch_size": 16
  }
  ```
  - `filenames`: List of image filenames (from the uploaded stack) to run inference on. Can be a single filename or multiple.
  - `batch_size` (optional): Number of images to process at once in each mini-batch. If not provided, the backend will auto-detect the optimal batch size for your system.
- Response: List of results for each filename, e.g.
  ```json
  [
    {"filename": "image1.png", "class": "cat", "confidence": 0.98},
    {"filename": "image2.png", "class": "dog", "confidence": 0.87}
  ]
  ```

**/predict_optimized**
- Endpoint: `POST /predict_optimized`
- Description: Run AI inference on multiple images using the optimized ResNet18 model with OpenVINO (CPU).
- **This is the endpoint used by the web app for all inference requests.**
- Request body (JSON):
  ```json
  {
    "filenames": ["image1.png", "image2.png"],
    "batch_size": 16
  }
  ```
  - `filenames`: List of image filenames (from the uploaded stack) to run inference on. Can be a single filename or multiple.
  - `batch_size` (optional): Number of images to process at once in each mini-batch. If not provided, the backend will auto-detect the optimal batch size for your system.
- Response: List of results for each filename, e.g.
  ```json
  [
    {"filename": "image1.png", "class": "cat", "confidence": 0.98},
    {"filename": "image2.png", "class": "dog", "confidence": 0.87}
  ]
  ```
  
## 🔧 Model Optimization: Tools & Steps

To accelerate inference and reduce CPU workload, the ResNet18 model used in this project was optimized using **Post-Training Quantization (PTQ)** and **OpenVINO inference acceleration**.

### 🛠️ Tools Used

- **PyTorch + Torchvision** – Original ResNet18 model  
- **NNCF (Neural Network Compression Framework)** – For INT8 quantization without retraining  
- **OpenVINO OVC (Model Optimizer)** – To convert and optimize the ONNX model for Intel CPUs  
- **OpenVINO Runtime** – For efficient batch inference on CPU  

---

### ⚙️ Optimization Steps

1. **Export Pretrained Model**  
   - Load `torchvision.models.resnet18(pretrained=True)`  
   - Save it to ONNX format using `torch.onnx.export()`

2. **Post-Training Quantization (PTQ)**  
   - Use **NNCF PTQ pipeline** to quantize the ONNX model to INT8 without retraining.  
   - Calibration performed on ~100–400 images from the actual dataset to maintain accuracy.

3. **Convert to OpenVINO IR (INT8)**  
   - Use **`ovc`** to convert the quantized ONNX model to OpenVINO IR:  
     ```bash
     ovc resnet18_int8.onnx --output_model openvino_model/resnet18_int8.xml --input "input[?,3,224,224]"
     ```
   - The `--input` flag enables **dynamic batching**, allowing flexible batch sizes during inference.

4. **Batch Inference with OpenVINO Runtime**  
   - Integrated OpenVINO runtime in the backend (`model.py`) to run optimized inference on batches of images.
   - Uses CPU-optimized kernels with vectorization (e.g., Intel VNNI instructions when available).

---

## 🚀 Benchmark Results

| Images | PyTorch FP32 | OpenVINO INT8 | Speedup |
|---------|--------------|----------------|---------|
| 50      | 4.344s       | 2.409s         | **1.8× faster** |
| 100     | 8.265s       | 4.778s         | **1.7× faster** |
| 200     | 16.6s        | 9.5s           | **1.75× faster** |
| 400     | 32.1s        | 18.8s          | **1.7× faster** |
| 600     | 51.5s        | 28.6s          | **1.8× faster** |

---

## 🖥️ Benchmark Environment

The benchmarks were executed on the following machine:

| Spec | Details |
|------|---------|
| **CPU** | Intel® Core™ i5-2450M CPU @ 2.50GHz (4 cores / 8 threads) |
| **RAM** | 8 GB |
| **OS** | Ubuntu 22.04.5 LTS |
| **Python** | 3.10.12 |
| **PyTorch** | 2.7.1 |
| **OpenVINO** | 2024.6.0 |
| **NNCF** | 2.17.0 |
| **Batch Size** | 8 |
---

### ✅ Outcome

- Reduced inference latency  
- Lower CPU usage  
- Maintained model accuracy within acceptable range  
- Enabled real-time batch processing in the web app

