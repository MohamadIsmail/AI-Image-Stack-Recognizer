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