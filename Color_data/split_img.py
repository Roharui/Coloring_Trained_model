import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as pimg

os.chdir("imgs")

def colors(code):
    img = np.array([code for i in range(100)])
    return img.reshape(10, 10, 3)
    

if __name__ == '__main__':

    img_name = "ddd.jpg"
    print(img_name)

    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = pimg.imread(img_name)

    img_shape = img.shape

    img = (img / 51).astype(np.uint8) * 51

    data = [tuple(x) for x in img.reshape(-1, 3)]
    unique, counts = np.unique(data, return_counts=True, axis=0)

    uc = np.concatenate([unique, counts.reshape(-1, 1)], axis=1)
    uc = uc[np.argsort(-counts)]

    print(img.dtype)
    cv2.imshow('imgs',img)
    cv2.waitKey(0)

    

    plt.subplot(2, 2, 1)
    plt.imshow(colors(uc[0, :-1]))

    plt.subplot(2, 2, 2)
    plt.imshow(colors(uc[1, :-1]))

    plt.subplot(2, 2, 3)
    plt.imshow(colors(uc[2, :-1]))

    plt.subplot(2, 2, 4)
    plt.imshow(colors(uc[3, :-1]))
    

    plt.show()
    
    #cv2.imwrite('test.jpeg', xxx.reshape(img_shape))
    
    