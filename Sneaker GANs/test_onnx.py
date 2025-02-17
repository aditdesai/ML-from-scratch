import onnxruntime as ort
import numpy as np
import torch
import matplotlib.pyplot as plt

session = ort.InferenceSession("wgan.onnx", providers=["CPUExecutionProvider"]) # loads model and prepares inference oon CPU

noise = torch.randn(1, 100, 1, 1).numpy().astype(np.float32)
# label = np.array([2], dtype=np.int64)

outputs = session.run(None, {"noise": noise})

fake_image = torch.tensor(outputs[0])
fake_image = fake_image.squeeze(0).numpy()
fake_image = (fake_image * 0.5) + 0.5
fake_image = np.transpose(fake_image, (1, 2, 0))

plt.imshow(fake_image)
plt.axis('off')
plt.show()