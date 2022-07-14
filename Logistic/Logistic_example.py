import torch
from torch import nn
from torch import optim
from torch.distributions import MultivariateNormal
import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data = self.linear(data)
        data = self.sigmoid(data)
        return data


mu1 = -3 * torch.ones(2)
mu2 = 3 * torch.ones(2)
sigma1 = torch.eye(2) * 0.5
sigma2 = torch.eye(2) * 2

m1 = MultivariateNormal(mu1, sigma1)
m2 = MultivariateNormal(mu2, sigma2)
x1 = m1.sample((100,))
x2 = m2.sample((100,))

y = torch.zeros((200, 1))
y[100:] = 1

x = torch.cat([x1, x2], dim=0)
idx = np.random.permutation(len(x))
x = x[idx]
y = y[idx]

lr_module = LogisticRegression()
loss = nn.BCELoss()
optimizer = optim.SGD(lr_module.parameters(), 0.03)
scores = lr_module(x)

batch_size = 10
liters = 10
for _ in range(liters):
    for i in range(int(len(x) / batch_size)):
        input = x[i * batch_size: (i + 1) * batch_size]
        target = y[i * batch_size: (i + 1) * batch_size]
        output = lr_module(input)
        optimizer.zero_grad()
        L = loss(output, target)
        L.backward()
        optimizer.step()

w = lr_module.linear.weight[0]
b = lr_module.linear.bias[0]


def draw_decision_boundary(w0, b0, x0):
    x11 = (-b0-w0[0]*x0)/w0[1]
    plt.plot(x0.detach().numpy(), x11.detach().numpy(), 'r')
    plt.scatter(x1.numpy()[:, 0], x1.numpy()[:, 1])
    plt.scatter(x2.numpy()[:, 0], x2.numpy()[:, 1])
    plt.show()


draw_decision_boundary(w, b, torch.linspace(x.min(), x.max(), 50))
