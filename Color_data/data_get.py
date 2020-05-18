
import pandas as pd
import numpy as np

def getData():
    filename = "filename.csv"
    dataset = np.array(pd.read_csv(filename))[:, 1:]
    dataset = dataset.reshape(-1, 4, 3)


    X = []

    for i in list(dataset):
        for num, j in enumerate(i):
            _ = getEX(list(i), num)
            _.insert(0, list(j))
            X.append(_)

    return np.array(X)
    

def getEX(data, num):
    if num == 0:
        return data[1:]
    elif len(data) - 1 == num:
        return data[:num]
    else:
        return data[:num] + data[num+1:]

    

if __name__ == '__main__':
    getData()