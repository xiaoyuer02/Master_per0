# *******
# 自动求导
# *******
# 总结：
# 1. 梯度下降都是沿着反梯度方向进行参数更新
# 2. 小批量随机梯度下降是深度学习默认的学习算法
# 3. 重要的两个超参数批量大小b和学习率α。批量太小，浪费计算资源；批量太大，内存消耗大。
#    学习率太小，下降速度太慢；学习率太大，容易震荡出localminimal
import torch

x = torch.arange(4.0)
print(f"x:{x}", x.shape)

x.requires_grad = True
print(f"x.grad:{x.grad}") # 默认为None，保留梯度

y = 2 * torch.dot(x, x)
print(f"y:{y}", y)
y.backward()
print(f"x.grad:{x.grad}")
print(x.grad == 4 * x)

x.grad.zero_() # pytorch会累加梯度，因此先对此清零
print(f"x.grad:{x.grad}")

y = sum(x)
print(f"y:{y}", y)
y.backward()
print(f"x.grad:{x.grad}")

x.grad.zero_()
y = x * x # 此处y为向量而非标量
print(f"y = {y}")
y.sum().backward() # 先做求和再反向传播 
print(f"x.grad:{x.grad}")

x.grad.zero_()
y = x * x
print(f"y:{y}")
u = y.detach() # detach后没有了grad_fn = <MulBackward0> u不再是x的函数
print(f"u:{u}")
z = u * x # u被当常数？
z.sum().backward() # 没有sum()转成标量的话 对向量会报错
print(f"x.grad:{x.grad}")
print(x.grad == u)
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

def f(a): # 即使构建函数的计算图需要通过Python控制流，仍然可以计算变量的梯度
    b = 2 * a # Pytorch的隐式构造对这种控制流效果更好，但也会更慢
    while b.norm() < 1000:
        b = b * 2
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
    return c

a = torch.randn(size = (), requires_grad = True)
d = f(a)
d.backward()
print(f"a:{a}, d:{d}")
print(f"a.grad:{a.grad}")
print(a.grad == d / a)

 