import numpy as np
import struct
import matplotlib.pyplot as plt


def read_training_set(kind='train'):
    with open("train-labels-idx1-ubyte", 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)

    with open("train-images-idx3-ubyte", 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)

    with open("t10k-labels-idx1-ubyte", 'rb') as tlbpath:
        magic, n = struct.unpack('>II', tlbpath.read(8))
        test_labels = np.fromfile(tlbpath,dtype=np.uint8)

    with open("t10k-images-idx3-ubyte", 'rb') as timgpath:
        magic, num, rows, cols = struct.unpack('>IIII',timgpath.read(16))
        test_images = np.fromfile(timgpath, dtype=np.uint8).reshape(len(test_labels), 784)

    # Divide all the pixel values by 255
    # Every picture is made of 28 * 28 pixel points, presented in an 1-D array of size 784
    return np.true_divide(images, 255), labels, np.true_divide(test_images, 255)


def show_image(Xs, w1, w2, b1, b2, index):
    fig, ax = plt.subplots(
        nrows=10,
        ncols=10,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    i = 0
    for inde in index:
        img = decode(w2,b2, encode(w1, b1, Xs[inde])).reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        i += 1

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def draw_graph(errors):
    x_axis = np.arange(len(errors))
    plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_axis, errors, ".")
    plt.show()

def show_org(Xs, index):
    fig, ax = plt.subplots(
        nrows=10,
        ncols=10,
        sharex=True,
        sharey=True, )

    ax = ax.flatten()
    i = 0
    for inde in index:
        img = Xs[inde].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        i += 1

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def save_weights(w1, w2, b1, b2):
    with open("results.txt", 'w') as f:
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open("results.txt", 'w') as f:
            f.write("W1 is : ")
            f.write(np.array2string(w1, separator=', '))
            f.write("W2 is : ")
            f.write(np.array2string(w2, separator=', '))
            f.write("b1 is : ")
            f.write(np.array2string(b1, separator=', '))
            f.write("b2 is : ")
            f.write(np.array2string(b2, separator=', '))


# Component functions for Auto Encoder
def transfer_function(x):
    return np.tanh(x)


def d_transfer_function(x):
    return 1 - (np.tanh(x)) ** 2


def get_log_loss(x, y):
    index = 0
    total_cost = 0
    while index < x.shape[0]:
        if y[index] <= 0 or y[index] >= 1:
            index += 1
            continue
        total_cost += (-x[index] * np.log(y[index]) - (1 - x[index]) * np.log(1 - y[index]))
        index += 1
    return total_cost


def encode(w1, b1, x):   # x is an nd-array of size 784*1, w1 is an nd-array of size 30*784, b1 of size 30*1
    return transfer_function(w1.dot(x) + b1)


def decode(w2, b2, h):   # h is an nd-array of size 30*1, w2 is an nd-array of size 784*30, b2 of size 784*1
    return transfer_function(w2.dot(h) + b2)


# I am using Mini-batch Gradient Descent in this case
def get_gradient(x_batch, w1, w2, b1, b2):
    batch_size = len(x_batch)
    total_cost = 0
    g_w1 = np.zeros(w1.shape)
    g_w2 = np.zeros(w2.shape)
    g_b1 = np.zeros(b1.shape)
    g_b2 = np.zeros(b2.shape)
    for x in x_batch:
        y = decode(w2, b2, encode(w1, b1, x))
        # Reverse mode AutoDiff schedule
        d1 = (y - x) * d_transfer_function(y)
        g_b2 += d1
        g_w2 += np.outer(d1, encode(w1, b1, x))
        d2 = (w2.T.dot(d1)) * d_transfer_function(encode(w1, b1, x))
        g_b1 += d2
        g_w1 += np.outer(d2, x)

        # Update cost
        total_cost += get_log_loss(x, y)
    # print total_cost / batch_size
    return g_w1/batch_size, g_w2/batch_size, g_b1/batch_size, g_b2/batch_size, total_cost/batch_size


class AdamOptimizer():
    def __init__(self):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.gamma = 10 ** (-8)
        self.t = 0
        self.step_size = 0.001
        self.m0 = 0     #First moment vector
        self.v0 = 0     #Second moment vector

    def optimize(self, gradient):
        self.t += 1
        self.m0 = self.beta1 * self.m0 + (1 - self.beta1) * gradient
        self.v0 = self.beta2 * self.v0 + (1 - self.beta2) * (gradient ** 2)
        m = self.m0 / (1 - self.beta1 ** self.t)
        v = self.v0 / (1 - self.beta2 ** self.t)
        optimized_grad = self.step_size * m / (v ** 0.5 + self.gamma)
        return optimized_grad


def train(w1, w2, b1, b2, Xs, errors):
    t = 0
    batch_size = 20
    w1_opt = AdamOptimizer()
    w2_opt = AdamOptimizer()
    b1_opt = AdamOptimizer()
    b2_opt = AdamOptimizer()

    while t < 1000:  # Run 1000 times
        batch_index = 0
        loss = 0
        while batch_index < len(Xs) / batch_size:  # Here run 60000/20 times
            batch = Xs[batch_index*batch_size: (batch_index+1)*batch_size]
            batch_index += 1
            g_w1, g_w2, g_b1, g_b2, cost = get_gradient(batch, w1, w2, b1, b2)
            w1 -= w1_opt.optimize(g_w1)
            w2 -= w2_opt.optimize(g_w2)
            b1 -= b1_opt.optimize(g_b1)
            b2 -= b2_opt.optimize(g_b2)
            errors.append(cost/batch_size)
            loss += cost
        print "t= ",t
        print "loss is: ", loss / (len(Xs) / batch_size)
        if t % 50 == 0:
            save_weights(w1, w2, b1, b2)
        if t == 2:
            draw_graph(errors)
        t += 1


def AutoEncoder(Xs):
    # Parameter initialization
    w1 = np.random.normal(0, 0.01, (30, 784))
    w2 = np.random.normal(0, 0.01, (784, 30))
    b1 = np.zeros(30)
    b2 = np.zeros(784)
    errors = []
    train(w1, w2, b1, b2, Xs, errors)


if __name__ == "__main__":
    images, labels, test_images = read_training_set()  # Training Set with size 60,000
    Xs = []     # A list of ndarrays, each ndarray with shape 784*1
    index = 0
    while index < 60000:
        Xs.append(images[index])
        index += 1
    AutoEncoder(Xs)
    #show_image(test_images, w1, w2, b1, b2, index)
    #show_org(test_images, index)