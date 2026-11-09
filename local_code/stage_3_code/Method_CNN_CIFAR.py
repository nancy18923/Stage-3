"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn


class Method_CNN_CIFAR(method, nn.Module):
    data = None

    # ---- data spec, must match what Dataset_Loader returns -----------
    in_channels = 3
    image_size = (32, 32)
    num_classes = 10

    # ---- CNN architecture: every knob below is for the ablation study --
    conv_channels = [32, 64, 128]      # one entry per conv block, depth = len(conv_channels)
    kernel_size = 3
    padding = 1                   # 3x3 kernel with padding=1 keeps the spatial size
    stride = 1
    pool_size = 2
    pool_type = 'max'             # 'max' / 'avg'
    adaptive_pool = (2, 2)        # final spatial size before flattening
    hidden_dims = [128]           # fully connected layers after the conv trunk
    dropout = 0.3                 # 360张训练图，必须重正则
    use_batch_norm = True

    # ---- training ----------------------------------------------------
    max_epoch = 20          # max rounds to train the model
    learning_rate = 1e-3    # learning rate for gradient descent
    batch_size = 64
    weight_decay = 0.0
    loss_func = 'ce'        # 'ce' / 'nll'
    optimizer_name = 'adam'  # 'adam' / 'sgd'
    device = 'cpu'          # 'cuda' if a GPU is available
    loss_history = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        pool_layer = nn.MaxPool2d if self.pool_type == 'max' else nn.AvgPool2d
        # conv trunk: [Conv -> BN -> ReLU -> Pool] x depth
        layers = []
        c_in = self.in_channels
        for c_out in self.conv_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            layers.append(pool_layer(self.pool_size))
            c_in = c_out

        # adaptive pooling makes the flatten dimension independent of the input
        # size, so the same code serves 28x28, 112x92 and 32x32 inputs
        layers.append(nn.AdaptiveAvgPool2d(self.adaptive_pool))
        layers.append(nn.Flatten())

        feat_dim = c_in * self.adaptive_pool[0] * self.adaptive_pool[1]
        for h in self.hidden_dims:
            layers.append(nn.Linear(feat_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            feat_dim = h
        layers.append(nn.Linear(feat_dim, self.num_classes))

        self.net = nn.Sequential(*layers)

    # define the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        """Forward propagation"""
        return self.net(x)

    # ---- helpers -----------------------------------------------------
    def _set_mode(self, training):
        # do NOT call self.eval(): nn.Module.eval() is implemented as self.train(False),
        # and self.train is overridden below with the signature (x, y)
        nn.Module.train(self, training)

    def _compute_loss(self, logits, target):
        if self.loss_func == 'nll':
            return nn.NLLLoss()(torch.log_softmax(logits, dim=1), target)
        return nn.CrossEntropyLoss()(logits, target)

    def _make_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate,
                                   momentum=0.9, weight_decay=self.weight_decay)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)

    def _evaluate(self, X, Y):
        self._set_mode(False)
        with torch.no_grad():
            logits = self.forward(X)
            loss = self._compute_loss(logits, Y).item()
            acc = (logits.argmax(dim=1) == Y).float().mean().item()
        self._set_mode(True)
        return loss, acc

    def train(self, x, y):
        X = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        optimizer = self._make_optimizer()

        # the test split is injected by Setting_Train_Test before run(),
        # so a test curve can be recorded next to the training curve
        test_X = test_Y = None
        if self.data is not None and 'test' in self.data:
            test_X = torch.as_tensor(self.data['test']['X'], dtype=torch.float32, device=self.device)
            test_Y = torch.as_tensor(self.data['test']['Y'], dtype=torch.long, device=self.device)

        self._set_mode(True)
        for epoch in range(self.max_epoch):
            perm = torch.randperm(len(X), device=self.device)
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = X[idx], Y[idx]

                logits = self.forward(xb)
                loss = self._compute_loss(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = len(idx)
                epoch_loss += loss.item() * bs
                epoch_correct += (logits.argmax(dim=1) == yb).sum().item()
                epoch_total += bs

            self.loss_history['train_loss'].append(epoch_loss / epoch_total)
            self.loss_history['train_acc'].append(epoch_correct / epoch_total)

            message = (f'epoch {epoch + 1:02d}/{self.max_epoch} | '
                  f'loss {epoch_loss / epoch_total:.4f} | '
                  f'train acc {epoch_correct / epoch_total:.4f} ')

            if test_X is not None:
                test_loss, test_acc = self._evaluate(test_X, test_Y)
                self.loss_history['test_loss'].append(test_loss)
                self.loss_history['test_acc'].append(test_acc)
                message += f' | test loss {test_loss:.4f} | test acc {test_acc:.4f}'

            print(message)

    def test(self, x):
        self._set_mode(False)  # 每折开始前把权重恢复成随机初始值
        with torch.no_grad():
            logits = self.forward(torch.as_tensor(x, dtype=torch.float32, device=self.device))
        # hand numpy arrays back so that sklearn metrics / pickle stay plain
        return logits.argmax(dim=1).cpu().numpy()

    def run(self):
        self.to(self.device)
        self.reset()  # 每折开始前把权重恢复成随机初始值
        self.loss_history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

        print(f'{self.method_name}: depth={len(self.conv_channels)}, channels={self.conv_channels}, '
              f'kernel={self.kernel_size}, padding={self.padding}, pool={self.pool_type}{self.pool_size}, '
              f'dropout={self.dropout}, input={self.in_channels}x{self.image_size}')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['Y'])
        print('--start testing...')
        pred_Y = self.test(self.data['test']['X'])

        return {'pred_Y': pred_Y, 'true_Y': self.data['test']['Y'], 'loss_history': self.loss_history}

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.reset_parameters()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.reset_running_stats()