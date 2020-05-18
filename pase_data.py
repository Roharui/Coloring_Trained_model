import numpy as np
import os

base_path = "./data"

def arrayNum(num):
    result = []
    for i in range(16):
        if i == num:
            result.append(1.0)
        else:
            result.append(0.0)
    return result

with open(base_path+"/colors.dat","r") as f:
    color = f.read().split("\n")
    colors = []
    num = []
    for i in color:
        ii = i.split(":")
        _ = []
        for j in ii[0].split(","):
            _.append(round(int(j)/ 255.0, 3))
        colors.append(_)
        num.append(arrayNum(int(ii[1])))

np.save("colors.npy", np.array(colors))
np.save("num.npy", np.array(num))

