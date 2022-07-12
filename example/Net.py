import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Model import *


# 准备数据集
train_data_set = torchvision.datasets.CIFAR10(root='dataset',
                                              train=True,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)
test_data_set = torchvision.datasets.CIFAR10(root='dataset',
                                             train=False,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)
train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

train_data_loader = DataLoader(train_data_set, batch_size=64)
test_data_loader = DataLoader(test_data_set, batch_size=64)

# 参数设定
learning_rate = 1e-2
epochs = 10
total_train_step = 0
total_test_step = 0

# 创建网络模型
net = Net()
net.cuda()
loss_fc = nn.CrossEntropyLoss()
loss_fc.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
writer = SummaryWriter('logs_GPU')

# 设置训练
for epoch in range(epochs):
    print("-------第{}轮训练开始--------".format(epoch+1))

    # 训练步骤开始
    for data in train_data_loader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = net(imgs)
        loss = loss_fc(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar('train_loss', loss, total_train_step)
            print("训练次数:{}, loss:{}".format(total_train_step, loss.item()))

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = net(imgs)
            loss = loss_fc(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    total_test_step += 1
    print("第{}轮训练测试集loss:{}".format(epoch+1, total_test_loss))
    print("第{}轮训练测试集正确率:{}".format(epoch+1, total_accuracy/test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('accuracy', total_accuracy/test_data_size, total_test_step)

    torch.save(net.state_dict(), 'Net_result_GPU/net_{}_epoch.pth'.format(epoch+1))
    print("模型已保存")
writer.close()
