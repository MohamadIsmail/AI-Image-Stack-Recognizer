import io
import zipfile
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def create_dummy_zip():
    # Create a dummy PNG file in memory
    import numpy as np
    from PIL import Image
    img_bytes = io.BytesIO()
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('test.png', img_bytes.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def test_upload_and_predict():
    # Upload a dummy zip
    zip_buffer = create_dummy_zip()
    response = client.post(
        "/upload",
        files={"file": ("test.zip", zip_buffer, "application/zip")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["num_images"] == 1
    assert data["filenames"] == ["test.png"]

    # Test /predict endpoint
    predict_resp = client.post(
        "/predict",
        json={"filenames": ["test.png"]}
    )
    assert predict_resp.status_code == 200
    preds = predict_resp.json()
    assert isinstance(preds, list)
    assert preds[0]["filename"] == "test.png"
    assert "class" in preds[0]
    assert "confidence" in preds[0]

    # Test /predict_optimized endpoint
    ov_resp = client.post(
        "/predict_optimized",
        json={"filenames": ["test.png"]}
    )
    assert ov_resp.status_code == 200
    ov_preds = ov_resp.json()
    assert isinstance(ov_preds, list)
    assert ov_preds[0]["filename"] == "test.png"
    assert "class" in ov_preds[0]
    assert "confidence" in ov_preds[0]

def test_upload_non_zip():
    # Try uploading a non-ZIP file
    resp = client.post(
        "/upload",
        files={"file": ("notazip.txt", b"hello world", "text/plain")}
    )
    assert resp.status_code == 400
    assert "Only .zip files are allowed" in resp.text

def test_upload_zip_no_images():
    # Create a zip file with no images
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr('readme.txt', b"not an image")
    zip_buffer.seek(0)
    resp = client.post(
        "/upload",
        files={"file": ("empty.zip", zip_buffer, "application/zip")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_images"] == 0
    assert data["filenames"] == []

def test_predict_empty_filenames():
    # Should return 400 for empty filenames
    resp = client.post(
        "/predict",
        json={"filenames": []}
    )
    assert resp.status_code == 400
    assert "No filenames provided" in resp.text

def test_predict_nonexistent_filename():
    # Should return error for non-existent file
    resp = client.post(
        "/predict",
        json={"filenames": ["doesnotexist.png"]}
    )
    assert resp.status_code == 200
    preds = resp.json()
    assert isinstance(preds, list)
    assert preds[0]["filename"] == "doesnotexist.png"
    assert preds[0]["class"] == "error"
    assert preds[0]["confidence"] == 0.0
    assert "File not found" in preds[0]["error"]

def test_predict_optimized_empty_filenames():
    # Should return 400 for empty filenames
    resp = client.post(
        "/predict_optimized",
        json={"filenames": []}
    )
    assert resp.status_code == 400
    assert "No filenames provided" in resp.text

def test_predict_optimized_nonexistent_filename():
    # Should return error for non-existent file
    resp = client.post(
        "/predict_optimized",
        json={"filenames": ["doesnotexist.png"]}
    )
    assert resp.status_code == 200
    preds = resp.json()
    assert isinstance(preds, list)
    assert preds[0]["filename"] == "doesnotexist.png"
    assert preds[0]["class"] == "error"
    assert preds[0]["confidence"] == 0.0
    assert "File not found" in preds[0]["error"] 