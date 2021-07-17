import numpy as np
from sklearn.neural_network import MLPClassifier
import sys


def write_predictions(fname, arr):
    np.savetxt(fname, arr, fmt="%d", delimiter="\n")


X_train_file = sys.argv[1]
Y_train_file = sys.argv[2]
X_test_file = sys.argv[3]
output_file = sys.argv[4]
batch_size = int(sys.argv[5])
hidden_layer_list = sys.argv[6]
activation_type = int(sys.argv[7])

x_train = np.load(X_train_file)
x_train = [np.array(x_t.flatten()) for x_t in x_train]
y_train = np.load(Y_train_file)
x_test = np.load(X_test_file)
x_test = [np.array(x_t.flatten()) for x_t in x_test]
hidden_layer_list = [int(st) for st in hidden_layer_list.split(' ')]

clf = MLPClassifier(max_iter=100, activation=activation_type, batch_size=batch_size, solver='sgd',
                    learning_rate='adaptive',
                    hidden_layer_sizes=hidden_layer_list)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
write_predictions(output_file, y_pred)
