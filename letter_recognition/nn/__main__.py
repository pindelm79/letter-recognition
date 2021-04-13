from torchvision import datasets
import numpy as np

import letter_recognition.nn.activation as activation
import letter_recognition.nn.layers as nn
import letter_recognition.nn.loss as loss

if __name__=="__main__":
    # Dataset import
    train = datasets.EMNIST("./letter_recognition/data/", train=True, split="letters", download=True)
    # test = datasets.EMNIST("./letter_recognition/data/", train=False, split="letters", download=False)

    # X-Y split
    images = np.empty((len(train), 1, 28, 28))
    labels = np.empty(len(train))
    for i in range(len(train)):
        images[i, 0] = train[i][0]
        labels[i] = train[i][1] - 1 # -1 because we index from 0 (A: 0, B: 1, etc...) 

    # Model hyperparameters
    conv1_out_channels = 8
    conv1_kernel_size = (3, 3)
    max_pool1_size = 2
    linear1_in_feat = conv1_out_channels * 13 * 13
    linear1_out_feat = 260
    lr = 0.01
    batch_size = 4

    # Model architecture
    conv1 = nn.Conv2d(1, conv1_out_channels, conv1_kernel_size)
    relu = activation.ReLU()
    maxpool1 = nn.MaxPool2d(max_pool1_size)
    linear1 = nn.Linear(linear1_in_feat, linear1_out_feat)
    linear2 = nn.Linear(linear1_out_feat, 26)
    criterion = loss.CrossEntropy()

    loss_sum = 0
    loss_print_period = 1024
    for i in range(0, images.shape[0] - batch_size, batch_size):
        # Mini-batch split
        x = images[i:i+batch_size]

        # Binarization
        x = np.where(x>128, 1.0, 0.0)
        
        # Forward pass
        out_conv1 = conv1.forward(x)
        out_relu1 = relu.forward(out_conv1)
        out_maxpool1, idx_maxpool1 = maxpool1.forward(out_relu1)
        out_maxpool1_reshaped = out_maxpool1.reshape(out_maxpool1.shape[0], linear1_in_feat)
        out_linear1 = linear1.forward(out_maxpool1_reshaped)
        out_relu2 = relu.forward(out_linear1)
        out_linear2 = linear2.forward(out_relu2)
        current_loss = criterion.calculate(out_linear2, labels[i:i+batch_size])

        # Gradient calculation
        dloss = criterion.backward(out_linear2, labels[i:i+batch_size])
        dx_linear2, dw_linear2, db_linear2 = linear2.backward(dloss, out_relu2)
        dx_relu2 = relu.backward(dx_linear2, out_linear1)
        dx_linear1, dw_linear1, db_linear1 = linear1.backward(dx_relu2, out_maxpool1_reshaped)
        dx_linear1 = dx_linear1.reshape(out_maxpool1.shape)
        dx_maxpool1 = maxpool1.backward(dx_linear1, out_relu1, idx_maxpool1)
        dx_relu1 = relu.backward(dx_maxpool1, out_conv1)
        _, dw_conv1, db_conv1 = conv1.backward(dx_relu1, x)

        # SGD
        conv1.weight -= lr * dw_conv1
        conv1.bias -= lr * db_conv1
        linear1.weight -= lr * dw_linear1
        linear1.bias -= lr * db_linear1
        linear2.weight -= lr * dw_linear2
        linear2.bias -= lr * db_linear2

        # Loss printing
        loss_sum += current_loss
        if i % loss_print_period == 0 and i != 0:
            print(f"{loss_print_period // batch_size}-batch avg loss: {loss_sum / (loss_print_period // batch_size)}")
            loss_sum = 0