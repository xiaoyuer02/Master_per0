# ****************
# 线性回归的从零实现
# ****************
import torch
import random
from d2l import torch as d2l

# 构造人造数据集
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。 """
    X = torch.normal(0, 1, (num_examples, len(w))) # mean = 0, std = 1, tuple of int or torch
    print(f"X.shape:{X.shape}")
    y = torch.matmul(X, w) + b # 通用乘法
    y += torch.normal(0, 0.01, y.shape)
    print(f"y.shape:{y.shape}, y_len:{len(y)}")
    return X, y.reshape(-1, 1) # 将一维向量reshape成二维张量

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) # 生成引入噪点的数据集

print("features: ", features[0])
print("labels: ", labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, 0].detach().numpy(), # 将带梯度追踪的张量requires_grad = True 从计算图中分离出为普通张量
                labels.detach().numpy(), 1) # detach()出来才能.numpy()
d2l.plt.show()

# 生成大小为batch_size的小批量
def batch_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) # 生成顺序列表
    random.shuffle(indices) # 随机打乱
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: i + min(batch_size, num_examples)]) # list转tensor
        yield features[batch_indices], labels[batch_indices] # 暂停

batch_size = 10
for X, y in batch_iter(batch_size, features, labels): 
    # print(X, "\n", y)
    break # 只取一次batch # 没有break会遍历完全num_examples 而且每次都是新的一次性生成器 顺序也会每次都会重新打乱

# 定义初始化参数
w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True)
b = torch.ones(1, requires_grad = True)

# 定义模型
def linreg(X, w, b):
    """ 线性回归模型 """
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """ 均方损失 """
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    """ 小批量随机梯度下降 """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in batch_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # l是形状为(batch_size, 1)的张量，进行求和为标量，进行求导
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}") # 默认:f为:.6f 保留六位小数

print(f"w的估计误差：{true_w - w.reshape(true_w.shape)}")
print(f"b的估计误差：{true_b - b}")


