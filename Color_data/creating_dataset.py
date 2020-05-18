import pandas as pd
import cv2
import numpy as np
import os

path = os.getcwd()
os.chdir('imgs')

class Dataset:
    def __init__(self):
        self.lisDir = os.listdir()
        self.data = []
        
    def getImg(self, img_name):
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = (img / 51).astype(np.uint8) * 51

        data = [tuple(x) for x in img.reshape(-1, 3)]
        unique, counts = np.unique(data, return_counts=True, axis=0)

        uc = unique[np.argsort(-counts)][:4]

        self.data.append(uc.reshape(12))

    def run(self):
        for i in self.lisDir:
            self.getImg(i)
    
    def save(self):
        df = pd.DataFrame(data=np.array(self.data), columns=['R1','B1','G1','R2','B2','G2','R3','B3','G3','R4','B4','G4'])
        os.chdir(path)
        df.to_csv("data.csv", mode='w')


if __name__ == '__main__':

    a = Dataset()

    a.run()
    a.save()


