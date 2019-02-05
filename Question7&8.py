import numpy as np
import matplotlib.pyplot as plt
import struct

def read_training_set(kind='train'):
    with open("train-labels-idx1-ubyte", 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open("train-images-idx3-ubyte", 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    with open("t10k-labels-idx1-ubyte", 'rb') as tlbpath:
        magic, n = struct.unpack('>II', tlbpath.read(8))
        test_labels = np.fromfile(tlbpath,dtype=np.uint8)

    with open("t10k-images-idx3-ubyte", 'rb') as timgpath:
        magic, num, rows, cols = struct.unpack('>IIII',timgpath.read(16))
        test_images = np.fromfile(timgpath, dtype=np.uint8).reshape(len(test_labels), 784)

    # Divide all the pixel values by 255
    # Every picture is made of 28 * 28 pixel points, presented in an 1-D array of size 784
    return np.true_divide(images, 255), labels, np.true_divide(test_images, 255)

def sqre_loss(x, y):
    z = x - y
    row = 0
    cost = 0
    while row < z.shape[0]:
        cost += 0.5 * (z[row]) ** 2
        row += 1
    return cost


def draw(squre_loss, train_images, test_images):
    if squre_loss == False :
        OptimisedW1 = np.array()  # I hided data here
        Optimisedb1 = np.array()  # I hided data here
        Optimisedb2 = np.array()  # I hided data here

    imgs = train_images[0:10]
    u1 = np.random.rand(30, 784)
    u2 = np.random.rand(30, 784)
    offset = 10
    f11_result_mat = np.zeros((offset, offset))
    f12_result_mat = np.zeros((offset, offset))
    f22_result_mat = np.zeros((offset, offset))

    z1_space = np.linspace(-0.5, 0.5, num=offset)
    z2_space = np.linspace(-0.5, 0.5, num=offset)

    z1_idx = np.arange(0, offset)
    z2_idx = np.arange(0, offset)

    for z1_i in z1_idx:
        for z2_i in z2_idx:
            loss_11 = 0
            loss_12 = 0
            loss_22 = 0
            z1 = z1_space[z1_i]
            z2 = z2_space[z2_i]
            for x in imgs:
                f11 = transfer_function(
                    Optimisedb2 + OptimisedW2.dot(transfer_function((Optimisedb1 + (OptimisedW1 + (z1 * u1) + (z2 * u2)).dot(x)))))
                f12 = transfer_function(
                    Optimisedb2 + (OptimisedW2 + (z2 * u2).T).dot(transfer_function((Optimisedb1 + (OptimisedW1 + (z1 * u1)).dot(x)))))
                f22 = transfer_function(
                    Optimisedb2 + (OptimisedW2 + (z1 * u1).T + (z2 * u2).T).dot(transfer_function((Optimisedb1 + OptimisedW1.dot(x)))))
                if squre_loss == True:
                    loss_11 += np.sum(sqre_loss(x, f11), axis=0)
                    loss_12 += np.sum(sqre_loss(x, f12), axis=0)
                    loss_22 += np.sum(sqre_loss(x, f22), axis=0)
                else:
                    loss_11 += np.sum(cross_loss(x, f11), axis=0)
                    loss_12 += np.sum(cross_loss(x, f12), axis=0)
                    loss_22 += np.sum(cross_loss(x, f22), axis=0)

            f11_result_mat[z1_i][z2_i] = loss_11 / 10
            f12_result_mat[z1_i][z2_i] = loss_12 / 10
            f22_result_mat[z1_i][z2_i] = loss_22 / 10

    plt.matshow(f11_result_mat)
    plt.colorbar()
    plt.show()


def transfer_function(x):
    return np.tanh(x)


def cross_loss(x, y):
    index = 0
    total_cost = 0
    while index < x.shape[0]:
        if y[index] <= 0 or y[index] >= 1:
            index += 1
            continue
        total_cost += (-x[index] * np.log(y[index]) - (1 - x[index]) * np.log(1 - y[index]))
        index += 1
    return total_cost


if __name__ =="__main__":
    images, labels, test_images = read_training_set()  # Training Set with size 60,000
    # square_loss = True
    # draw(square_loss, images, test_images)
    square_loss = False
    draw(False, images, test_images)
