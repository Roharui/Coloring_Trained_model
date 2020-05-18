
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape, ReLU
from data_get import getData
import numpy as np

class CRM:
    def __init__(self, epoc, batch_size):
        self.epoc = epoc
        self.batch_size = batch_size
    def train_and_test(self, ratio):
        dataset = getData()
        np.random.shuffle(dataset)

        dataset = dataset - min(dataset)
        dataset = dataset/max(dataset)

        div_number = int(len(dataset) / 10 * ratio)

        x_train = dataset[:div_number, -1].reshape(-1, 1, 3)
        y_train = dataset[:div_number, :-1]

        x_test = dataset[div_number:, -1]
        y_test = dataset[div_number:, :-1]



        self.dataset =  (x_train, y_train, x_test, y_test)


    def Model(self):
        model = Sequential()

        model.add(Dense(9, input_shape=(1,3)))
        model.add(ReLU())
        model.add(Dense(18))
        model.add(ReLU())
        model.add(Dense(9))
        model.add(ReLU())
        model.add(Reshape(3,3))

        model.summary()
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        self.model = model

    def train(self):
        x_train, y_train, x_test, y_test = self.dataset

        self.model.fit(x_train, y_train, epochs=self.epoc, batch_size=self.batch_size)


    def save_model(self):
        self.model.save('CRM.h5')

    def test(self):
        x_train, y_train, x_test, y_test = self.dataset

        loss_and_metrics = self.model.evaluate(x_test, y_test, batch_size=32)
        print(loss_and_metrics)

def main(epoc, batch_size):
    crm = CRM(epoc, batch_size)
    crm.train_and_test(8)
    crm.Model()
    crm.train()
    crm.save_model()


if __name__ == '__main__':
    a = CRM(10, 10)
    a.train_and_test(8)