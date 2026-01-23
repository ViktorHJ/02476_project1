from io import BytesIO
import random

from locust import HttpUser, between, task
from PIL import Image


class CIFakeUser(HttpUser):
    wait_time = between(1, 2)

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(4)
    def predict(self):
        # Create a valid 32x32 RGB image and encode as JPEG
        img = Image.new(
            "RGB",
            (32, 32),
            color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        )
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)

        files = {
            "data": ("test_image.jpg", buf, "image/jpeg"),
        }
        self.client.post("/predict/", files=files)