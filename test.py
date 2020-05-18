#from keras.models import Sequential, load_model
import numpy as np

#model = load_model("color.h5")

a = np.load("colors.npy")
c = np.load("num.npy")

print(c)

'''
result = 0

for i in range(100):
    n = randint(0, len(a))
    y = model.predict(a[n:n+1])
    if np.argmax(y[0]) == np.argmax(c[n]):
        result+= 1

print(result)
'''
