import numpy as np

print("y_test")
np.save("y_test.npy", np.array(np.load("y_test.npy"), dtype=np.float)[1:])

print("y_train")
np.save("y_train.npy", np.array(np.load("y_train.npy"), dtype=np.float)[1:])

print("x_test")
np.save("x_test.npy", np.array(np.load("x_test.npy"), dtype=np.float)[1:])

print("x_train")
np.save("x_train.npy", np.array(np.load("x_train.npy"), dtype=np.float)[1:])
