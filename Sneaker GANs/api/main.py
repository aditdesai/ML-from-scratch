from fastapi import FastAPI, Response
import onnxruntime as ort
import numpy as np
import torch
import cv2

app = FastAPI()
session_wgan = ort.InferenceSession("wgan.onnx", providers=['CPUExecutionProvider'])
session_cgan = ort.InferenceSession("cgan.onnx", providers=['CPUExecutionProvider'])

def to_bytes(img):
    img = (img * 255).astype(np.uint8)
    _, img_encoded = cv2.imencode(".png", img) # converts numpy array into specified format and returns it as a byte array

    return img_encoded.tobytes()


@app.get('/run_wgan')
def run_wgan():
    noise = torch.randn(1, 100, 1, 1).numpy().astype(np.float32)
    output = session_wgan.run(None, {"noise": noise})

    fake_image = torch.tensor(output[0])
    fake_image = fake_image.squeeze(0).numpy()
    fake_image = (fake_image * 0.5) + 0.5
    fake_image = np.transpose(fake_image, (1, 2, 0))

    return Response(content=to_bytes(fake_image), media_type='image/png')

@app.get('/run_cgan')
def run_cgan(label: int):
    noise = torch.randn(1, 100, 1, 1).numpy().astype(np.float32)
    label = np.array([label], dtype=np.int64)
    output = session_cgan.run(None, {"noise": noise, "label": label})

    fake_image = torch.tensor(output[0])
    fake_image = fake_image.squeeze(0).numpy()
    fake_image = (fake_image * 0.5) + 0.5
    fake_image = np.transpose(fake_image, (1, 2, 0))

    return Response(content=to_bytes(fake_image), media_type='image/png')