# ****************
# 线性回归的简洁实现
# ****************
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train = True):
    """ 构造一个pytorch数据迭代器 """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

# 模型的定义
from torch import nn
net = nn.Sequential(nn.Linear(2, 1)) # Sequential容器 list of layers

# 参数的初始化
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 损失函数
# 均方误差，也称为平方L2范数
loss = nn.MSELoss()

# 训练模块
# 实例化SGD
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad() # 清除grad
        l.backward() # 默认做了sum()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

print("w的偏差估计：", (true_w - net[0].weight.reshape(true_w.shape)))
print("b的偏差估计：", (true_b - net[0].bias.data))
