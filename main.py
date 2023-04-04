from perceptron import Perceptron
import pandas
import openpyxl
import numpy as np

if __name__ == '__main__':
    p = Perceptron(inputs=9, hidden=4, outputs=1)

    training_data = pandas.read_excel('1.xlsx')
    df = training_data.output.to_numpy()
    target_output = []
    for x in df:
        target_output.append([])
        target_output[len(target_output)-1].append(x)
    training_data = training_data.drop(['output'], axis=1)
    training_data = training_data.drop(['comment'], axis=1)
    training_data = np.asarray(training_data)



    p.learn(training_data, target_output, LR=0.1, epoch=10000)

    validation_data = pandas.read_excel('2.xlsx')
    validation_data = validation_data.drop(['output'], axis=1)
    validation_data = validation_data.drop(['comment'], axis=1)
    validation_data = np.asarray(validation_data)

    prediction = p.predict(validation_data)
    print(prediction)
    for out in prediction:
        if out >= 0.5:
            print("1")
        else:
            print("0")
