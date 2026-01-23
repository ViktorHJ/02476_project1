from fastapi.testclient import TestClient
from cifakeclassification.api import app
from pathlib import Path
from io import BytesIO
from PIL import Image

client = TestClient(app)

HERE = Path(__file__).resolve().parent
TEST_IMG = HERE / "test_image.jpg"  # put the image next to this test file

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "device" in data
    assert "checkpoint" in data

def test_predict():
    # Create a tiny RGB image in memory
    buf = BytesIO()
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(buf, format="JPEG")
    buf.seek(0)

    with TestClient(app) as client:
        response = client.post(
            "/predict/",
            files={"data": ("test_image.jpg", buf, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["FAKE", "REAL"]
    assert "FAKE" in data["probs"]
    assert "REAL" in data["probs"]
    assert 0.0 <= data["probs"]["FAKE"] <= 1.0
    assert 0.0 <= data["probs"]["REAL"] <= 1.0