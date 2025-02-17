import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2

response = requests.get(f'http://127.0.0.1:8000/run_wgan')
if response.status_code == 200:
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is not None:
        plt.imshow(img)
        plt.axis('off')
        plt.show()