'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    hidden_dims = [256]
    # it defines the max rounds to train the model
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 64
    loss_func = 'ce'            # 'ce' / 'nll'
    optimizer_name = 'adam'     # 'adam' / 'sgd'
    loss_history = None
    train_acc_history = None

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        dims = [784] + list(self.hidden_dims) + [10]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))        # 输出层，不加激活
        self.net = nn.Sequential(*layers)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # self.activation_func_2 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        return self.net(x)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, Y):
        X = torch.as_tensor(X, dtype=torch.float32)
        Y = torch.as_tensor(Y, dtype=torch.long)
        self.loss_history = []
        self.train_acc_history = []

        if self.loss_func == 'nll':
            loss_function = None
            compute_loss = lambda z, t: nn.NLLLoss()(torch.log_softmax(z, dim=1), t)
        else:
            compute_loss = nn.CrossEntropyLoss()

        if self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            perm = torch.randperm(len(X))
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = X[idx], Y[idx]

                logits = self.forward(xb)
                # calculate the training loss
                loss = compute_loss(logits, yb)
                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = len(idx)
                epoch_loss += loss.item() * bs
                epoch_correct += (logits.argmax(1) == yb).sum().item()
                epoch_total += bs

            self.loss_history.append(epoch_loss / epoch_total)
            self.train_acc_history.append(epoch_correct / epoch_total)

            print(f'epoch {epoch + 1:02d}/{self.max_epoch} | '
                  f'loss {epoch_loss / epoch_total:.4f} | '
                  f'train acc {epoch_correct / epoch_total:.4f} ')

    def test(self, X):
        with torch.no_grad():
            # do the testing, and result the result
            y_pred = self.forward(torch.as_tensor(X, dtype=torch.float32))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        self.reset()            # 每折开始前把权重恢复成随机初始值
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['Y'])
        print('--start testing...')
        pred_Y = self.test(self.data['test']['X'])
        return {'pred_Y': pred_Y, 'true_Y': self.data['test']['Y'],
                'loss_history': self.loss_history,
                'train_acc_history': self.train_acc_history}

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()