from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

colors = np.load("colors.npy")
num = np.load("num.npy")

colors = colors.reshape((100, 3,))
num = num.reshape((100, 16,))

x_train = colors[:80]
y_train = num[:80]

x_test = colors[80:]
y_test = num[80:]

model = Sequential()

model.add(Dense(64, input_dim=3, activation="relu"))
model.add(Dense(64, input_dim=64, activation="relu"))
model.add(Dense(16, input_dim=64, activation="softmax"))

model.compile(metrics=['accuracy'], optimizer="adam", loss='categorical_crossentropy')

es = EarlyStopping()
hist = model.fit(x_train, y_train, epochs=200, batch_size=16, validation_data=(x_test, y_test), callbacks=[es])

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

model.save("color.h5")

scores = model.evaluate(x_test, y_test, batch_size=16)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
