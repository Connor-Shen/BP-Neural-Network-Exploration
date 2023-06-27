import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import cv2
import pickle

dic = loadmat("D:/desktop/NN&DL/project_1_release/codes/digits.mat")
# idx = 125
# img = dic["Xvalid"][idx, :].reshape(16,16)
# print(dic["yvalid"][idx])
# plt.imshow(img,cmap="gray")
# plt.show()

def sigmoid_func(x):  # 解决了overflow问题后显著提高正确率
    return .5 * (1 + np.tanh(.5 * x))

def softmax(x):
    exps = np.exp(x- np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)

def negative_log_likelihood(self, y_true, y_pred):
    return -np.mean(np.log(y_pred[range(len(y_true)), y_true]))

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def cross_entropy_loss(prediction, label):
    num_classes = prediction.shape[0]
    label_vector = np.zeros((num_classes, 1))
    label_vector[label] = 1
    loss = -np.sum(label_vector * np.log(prediction))
    return loss

def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability matrix corresponding to your label predictions, shape (classes, number of examples)
    Y -- true "label" matrix, shape (classes, number of examples)

    Returns:
    cost -- softmax cross-entropy cost
    """
    m = Y.shape[0]
    cost = -(np.sum(Y * np.log(AL))) / float(m)
    # cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost


pkl_file = open('trained_wei4.pkl', 'rb')

# 从文件中读取
trained_weight = pickle.load(pkl_file)
pkl_file.close()


class NeuralNetwork:
    # initialise the neural network
    # 对神经网络的参数进行初始化
    def __init__(self, innodes, hid1nodes, hid2nodes, outnodes, learningrate, trained_weight):  # 用于类的初始值的设定
        # set number of nodes in each input, hidden, output layer
        # 设置节输入层、隐藏层、输出层的节点数  （self表示类所自带的不能改变的内容）
        self.inodes = innodes  # 输入层节点数
        self.h1nodes = hid1nodes  # 隐藏层1节点数
        self.h2nodes = hid2nodes  # 隐藏层2节点数
        self.onodes = outnodes  # 输出层节点数

        self.w1 = trained_weight[0]
        self.w2 = trained_weight[1]
        self.b1 = trained_weight[2]
        self.b2 = trained_weight[3]
        # link weight matrices, wih and who
        # 设置输入层与隐藏层直接的权重关系矩阵以及隐藏层与输出层之间的权重关系矩阵
        # （一开始随机生成权重矩阵的数值，利用正态分布，均值为0，方差为隐藏层节点数的-0.5次方，）
        # self.w1 = np.random.normal(0.0, pow(self.h1nodes, -0.5), (self.h1nodes, self.inodes))  # 矩阵大小为隐藏层1节点数×输入层节点数
        # self.w2 = np.random.normal(0.0, pow(self.h1nodes, -0.5), (self.h2nodes, self.h1nodes))  # 矩阵大小为隐藏层2节点数×隐藏层1节点数
        self.w3 = np.random.normal(0.0, pow(self.h2nodes, -0.5), (self.onodes, self.h2nodes))  # 矩阵大小为输出层节点数×隐藏层2节点数
        # self.b1 = np.zeros((self.h1nodes, ))
        # self.b2 = np.zeros((self.h2nodes,))
        self.b3 = np.zeros((self.onodes, ))


        # self.V_dW1 = np.zeros_like(self.w1)
        # self.V_db1 = np.zeros_like(self.b1)
        # self.V_dW2 = np.zeros_like(self.w2)
        # self.V_db2 = np.zeros_like(self.b2)
        self.V_dW3 = np.zeros_like(self.w3)
        self.V_db3 = np.zeros_like(self.b3)

        self.v1 = 0
        self.v2 = 0
        self.v3 = 0
        # learning rate
        # 设置学习率α
        self.lr = learningrate
        self.momentum = momentum
        self.lamda = lamda

        # activation function is the sigmoid function
        # 将激活函数sigmoid定义为self.activation_function
        self.activation_function = lambda x: sigmoid_func(x)
        # lambda x:表示快速生成函数f(x) 并将其命名为self.activation_function

        pass

    def train(self, inputs, targets):
        # w1->(100*256), w2->(10*100)
        # Forward Propagation
        hidden_inputs_1 = np.dot(self.w1, inputs) + self.b1 # dot()函数是指两个矩阵做点乘  (100*1)
        # calculate the signals emerging from hidden layer
        # using sigmoid
        hidden_outputs_1 = (sigmoid_func(hidden_inputs_1))  # (100*1)
        # add dropout layer
        drop_out1 = np.random.rand(*hidden_outputs_1.shape) < p
        hidden_outputs_1 *= drop_out1

        hidden_inputs_2 = np.dot(self.w2, hidden_outputs_1) + self.b2
        hidden_outputs_2 = (sigmoid_func(hidden_inputs_2))
        drop_out2 = np.random.rand(*hidden_outputs_2.shape) < p
        hidden_outputs_2 *= drop_out2

        final_inputs = np.dot(self.w3, hidden_outputs_2) + self.b3# (10*1)
        # calculate the signals emerging from final output layer
        final_outputs = (sigmoid_func(final_inputs))  # (10*1)

        # Backward Propagation
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs  # (10*1)
        # delta represents the error term in bp algorithm
        # Derivatives of softmax is the form of "x(1-x)"
        output_delta = output_errors * final_outputs * (1.0 - final_outputs) # (10*1)^3 -->(10*1)
        # sum of w1 * the delta of next layer
        hidden_error_2 = np.dot(self.w3.T, output_delta) # (100*1)
        hidden_delta_2 = hidden_error_2 * hidden_outputs_2 * (1.0 - hidden_outputs_2)  # (100*1)

        hidden_error_1 = np.dot(self.w2.T, hidden_delta_2)  # (100*1)
        hidden_delta_1 = hidden_error_1 * hidden_outputs_1 * (1.0 - hidden_outputs_1)  # (100*1)

        # """softmax"""
        # grad_ho = np.dot(y_hat.reshape(-1,1), hidden_outputs.reshape(1,-1))
        # grad_ih = np.dot(hidden_delta.reshape(-1, 1), np.transpose(inputs.reshape(-1, 1)))
        # grad_bias_o = np.mean(y_hat, axis=0, keepdims=True)
        # grad_bias_h = np.mean(hidden_delta, axis=0, keepdims=True)
        # self.V_dW1 = self.momentum * self.V_dW1 + (1 - self.momentum) * grad_ih # (100*256)
        # self.V_db1 = self.momentum * self.V_db1 + (1 - self.momentum) * grad_bias_h
        # self.V_dW2 = self.momentum * self.V_dW2 + (1 - self.momentum) * grad_ho # (10*100)
        # self.V_db2 = self.momentum * self.V_db2 + (1 - self.momentum) * grad_bias_o

        dw3 = np.dot(output_delta.reshape(-1, 1), np.transpose(hidden_outputs_2.reshape(-1, 1)))
        # dw2 = np.dot(hidden_delta_2.reshape(-1, 1), np.transpose(hidden_outputs_1.reshape(-1, 1)))
        # # dw2 = np.dot(output_delta.reshape(-1, 1), np.transpose(hidden_outputs.reshape(-1, 1)))
        # dw1 = np.dot(hidden_delta_1.reshape(-1, 1), np.transpose(inputs.reshape(-1, 1)))
        db3 = np.sum(output_delta, axis=0, keepdims=True)
        # db2 = np.sum(hidden_delta_2, axis=0)
        # db1 = np.sum(hidden_delta_1, axis=0)

        # v表示x要改变的幅度
        # self.V_dW1 = self.momentum * self.V_dW1 + (1 - self.momentum) * dw1 # (100*256)
        # self.V_db1 = self.momentum * self.V_db1 + (1 - self.momentum) * db1
        # self.V_dW2 = self.momentum * self.V_dW2 + (1 - self.momentum) * dw2 # (10*100)
        # self.V_db2 = self.momentum * self.V_db2 + (1 - self.momentum) * db2
        self.V_dW3 = self.momentum * self.V_dW3 + (1 - self.momentum) * dw3 # (10*100)
        self.V_db3 = self.momentum * self.V_db3 + (1 - self.momentum) * db3

        # self.w1 += self.lr * (self.V_dW1 - self.lamda * self.w1) # lamda -> weight decay
        # self.b1 += self.lr * self.V_db1
        # self.w2 += self.lr * (self.V_dW2 - self.lamda * self.w2)
        # self.b2 += self.lr * self.V_db2
        self.w3 += self.lr * (self.V_dW3 - self.lamda * self.w3)
        self.b3 += self.lr * self.V_db3

        # self.w1 = self.w1 * (1 - self.lr * lamda) + dw1 * self.lr
        # self.w2 = self.w2 * (1 - self.lr * lamda) + dw2 * self.lr
        pass

    def weights(self):
        return [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]

    def weights_optim(self, best_w1, best_w2, best_w3, best_b1, best_b2, best_b3):
        self.w1 = best_w1
        self.w2 = best_w2
        self.w3 = best_w3
        self.b1 = best_b1
        self.b2 = best_b2
        self.b3 = best_b3

    def pre(self, inputs):
        hidden_inputs_1 = np.dot(self.w1, inputs) + self.b1  # dot()函数是指两个矩阵做点乘  (100*1)
        # calculate the signals emerging from hidden layer
        # using sigmoid
        hidden_outputs_1 = (sigmoid_func(hidden_inputs_1)) * p  # (100*1)

        hidden_inputs_2 = np.dot(self.w2, hidden_outputs_1) + self.b2
        hidden_outputs_2 = (sigmoid_func(hidden_inputs_2)) * p

        final_inputs = np.dot(self.w3, hidden_outputs_2) + self.b3  # (10*1)
        # calculate the signals emerging from final output layer
        final_outputs = (sigmoid_func(final_inputs))  # (10*1)

        return final_outputs


innodes = 256
hid1nodes = 180
hid2nodes = 125
outnodes = 10

learningrate = 0.3
momentum = 0.9
lamda = 0
patience = 5
p = 0.5

n = NeuralNetwork(innodes, hid1nodes, hid2nodes, outnodes, learningrate, trained_weight)

tr_data = dic["X"]
valid_data = dic["Xvalid"]

epochs = 50
score = 0
best_loss = 1
best_acu = 0



tr_data_rotation = []
for idx, data in enumerate(tr_data):
    img = data.reshape(16,16)
    if 0<= idx <= 200:
        M = cv2.getRotationMatrix2D((8, 8), 90, 1.0)
        rotation_img = cv2.warpAffine(img, M, (16, 16))
        tr_data_rotation.append(rotation_img.reshape(256,))
    elif 200< idx <= 500:
        M = cv2.getRotationMatrix2D((8, 8), 180, 1.0)
        rotation_img = cv2.warpAffine(img, M, (16, 16))
        tr_data_rotation.append(rotation_img.reshape(256,))
    elif 500< idx <= 700:
        M = cv2.getRotationMatrix2D((8, 8), 45, 1.0)
        rotation_img = cv2.warpAffine(img, M, (16, 16))
        tr_data_rotation.append(rotation_img.reshape(256,))
    elif 700< idx <= 1000:
        M = cv2.getRotationMatrix2D((8, 8), 270, 1.0)
        rotation_img = cv2.warpAffine(img, M, (16, 16))
        tr_data_rotation.append(rotation_img.reshape(256, ))
    else:
        break

## fine-tuning

trained_weight = []
final_input = []

for i in range(epochs):
    train_score = 0
    valid_score = 0
    tr_loss = 0
    valid_loss = 0
    for idx, data in enumerate(tr_data):
        inputs = (np.asfarray(data) / 255.0) * 0.99 + 0.01
        targets = np.zeros(outnodes) + 0.01
        targets[int(dic["y"][idx]-1)] = 0.99
        n.train(inputs, targets)
        outputs = n.pre(inputs)
        label = np.argmax(outputs)
        if (label + 1 == dic["y"][idx]):
            train_score += 1
        else:
            train_score += 0
        tr_loss += compute_cost(outputs, targets)
    for idx, data in enumerate(tr_data_rotation):
        inputs = (np.asfarray(data) / 255.0) * 0.99 + 0.01
        targets = np.zeros(outnodes) + 0.01
        targets[int(dic["y"][idx] - 1)] = 0.99
        n.train(inputs, targets)
        outputs = n.pre(inputs)
        label = np.argmax(outputs)
        if (label + 1 == dic["y"][idx]):
            train_score += 1
        else:
            train_score += 0
        tr_loss += compute_cost(outputs, targets)
    # tr_loss = tr_loss / (len(tr_data))
    tr_loss = tr_loss/(len(tr_data)+len(tr_data_rotation))
    # tr_accuracy = train_score / (len(tr_data))
    tr_accuracy = train_score/(len(tr_data)+len(tr_data_rotation))
    for idx, data in enumerate(valid_data):
        data = (np.asfarray(data) / 255.0) * 0.99 + 0.01
        targets = np.zeros(outnodes) + 0.01
        targets[int(dic["yvalid"][idx] - 1)] = 0.99
        outputs = n.pre(data)
        label = np.argmax(outputs)  # argmax()函数用于找出数值最大的值所对应的标签
        if (label + 1 == dic["yvalid"][idx]):
            valid_score += 1
        else:
            valid_score += 0
        valid_loss += compute_cost(outputs, targets)
    valid_loss = valid_loss / len(dic["yvalid"])
    val_accuracy = valid_score/len(dic["yvalid"])
    print(f"Train accuracy: {tr_accuracy:.4f}. Valid accuracy: {val_accuracy:.4f}. Train loss: {tr_loss:.4f}. Valid loss: {valid_loss:.4f}")
    if val_accuracy > best_acu:
    # if valid_loss < best_loss:
    #     best_loss = valid_loss
        best_acu = val_accuracy
        best_w1 = n.weights()[0].copy()
        best_w2 = n.weights()[1].copy()
        best_w3 = n.weights()[2].copy()
        best_b1 = n.weights()[3].copy()
        best_b2 = n.weights()[4].copy()
        best_b3 = n.weights()[5].copy()
        stopping_rounds = 0
    else:
        stopping_rounds += 1
    if stopping_rounds >= patience:
        print(f"Validation accuracy has not improved for {patience} rounds. Stopping training.")
        n.weights_optim(best_w1, best_w2, best_w3, best_b1, best_b2, best_b3)
        trained_weight.append(best_w1)
        trained_weight.append(best_w2)
        trained_weight.append(best_b1)
        trained_weight.append(best_b2)
        break


te_data = dic["Xtest"]

te_lost = 0
for idx, data in enumerate(te_data):
    data = (np.asfarray(data) / 255.0) * 0.99 + 0.01
    targets = np.zeros(outnodes) + 0.01
    targets[int(dic["ytest"][idx] - 1)] = 0.99
    outputs = n.pre(data)
    label = np.argmax(outputs) # argmax()函数用于找出数值最大的值所对应的标签
    if (label+1 == dic["ytest"][idx]):
        score += 1
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        score += 0
    te_lost += compute_cost(outputs, targets)
te_accuracy = score / len(dic["ytest"])
te_lost = te_lost / len(dic["ytest"])
# calculate the performance score, the fraction
print(f"test accuracy : {te_accuracy: .4f} , test lost : {te_lost:.4f}")
