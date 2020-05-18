
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from random import randint

def cimg(color):
    return np.array([color for i in range(2500)]).reshape(50, 50, 3)

model = load_model("CRM.h5")

while True:

    img = np.array([[randint(0, 255), randint(0,255), randint(0, 255)]])

    x = model.predict(img / 255)

    y = (np.array(x) * 255).astype(np.uint8)

    plt.subplot(1, 4, 1)
    plt.imshow(cimg(img))

    for num, i in enumerate(y[0]):
        plt.subplot(1, 4, num + 2)
        plt.imshow(cimg(i))

    plt.show()