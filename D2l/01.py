import torch
import numpy as np

x = torch.arange(12) # 创建一个包含12个元素的张量

print(x) # 打印张量
print(x.reshape([3, 4])) # 打印重塑后的张量
print(x.reshape(3, 4)) # 打印重塑后的张量
print(x.shape) # 返回张量的形状
print(x.numel()) # 返回张量中元素的数量

print(torch.ones((2, 3, 4))) # 创建一个形状为(2, 3, 4)的全1张量
print(torch.zeros(2, 3, 4)) # 创建一个形状为(2, 3, 4)的全0张量

print(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])) # 从列表创建张量
print(torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]).shape) # 再在外面加一层列表，创建一个形状为(1, 3, 4)的张量，并打印其形状

x = torch.tensor([1.0, 2, 4, 8]) # 1.0是浮点数，运算后结果会是浮点数
y = torch.tensor([2  , 2, 2, 2]) # 整数
print(x + y, x - y, x * y, x / y, x ** y) # 张量的加减乘除和幂运算

print(torch.exp(x)) # 对张量中每个元素求指数

x = torch.arange(12, dtype = torch.float32).reshape(3, 4)
y = torch.tensor([[2, 1, 4, 3], [3, 2, 1, 4], [4, 3, 2, 1]])
print(torch.cat((x, y), dim = 0)) # 在第0个维度上连接张量 竖直
print(torch.cat((x, y), dim = 1)) # 在第1个维度上连接张量 水平

print(x==y) # 比较两个张量的每个元素是否相等，返回一个布尔张量 x,y形状要相同

print(x.sum()) # 求张量中所有元素的和

a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2) # 同样是2-D向量
print(f"a: {a}")
print(f"b: {b}")
print(a + b) # 广播机制：a的形状为(3, 1)，b的形状为(1, 2)，通过广播机制，a会扩展为(3, 2)，b会扩展为(3, 2)，然后进行逐元素相加，得到一个形状为(3, 2)的张量
# 虽然方便但有可能成为隐藏的坑

print(x[-1], x[1:3], x[1:]) # 访问张量的元素，x[-1]表示访问最后一行，x[1:3]表示访问第2行到第3行（不包括第3行），x[1:]表示访问第2行到最后一行

x[1, 2] = 9 # 修改张量中指定位置的元素，x[1, 2]表示访问第2行第3列的元素，将其修改为9
print(x) # 打印修改后的张量
x[0:2, :] = 12 # 修改张量中指定范围的元素，x[0:2, :]表示访问第1行到第2行的所有列，将其修改为12
print(x) # 打印修改后的张量

before = id(y) # 获取张量y的内存地址
y = y + x # 将y和x相加，并将结果赋值给y，这会创建一个新的张量对象，y的内存地址会发生变化
print(id(y) == before) # 打印y的内存地址是否发生变化，结果为False，说明y的内存地址发生了变化

z = torch.zeros_like(y) # 创建一个与y形状相同的全0张量z
before = id(z) # 获取张量z的内存地址
z[:] = y + x # 将y和x相加的结果赋值给z，这不会创建新的张量对象，而是将结果写入到z中，z的内存地址不会发生变化
print(id(z) == before) # 打印z的内存地址是否发生变化，结果为True，说明z的内存地址没有发生变化

before = id(y)
y[:] = z + x # 将z和x相加的结果赋值给y，这不会创建新的张量对象，而是将结果写入到y中，y的内存地址不会发生变化
print(id(y) == before) 

before = id(x)
x += y # 和上面的操作类似，x += y会将y加到x上，并将结果写入到x中，这不会创建新的张量对象，x的内存地址不会发生变化
print(id(x) == before) 

a = x.numpy() # 将张量转换为NumPy数组
b = torch.tensor(a) # 将NumPy数组转换为张量
print(a, type(a), b, type(b))

a = torch.tensor([3.5]) # 只有1个元素的张量才能转换为python标量
print(a, a.item(), float(a), int(a))
print(a.numpy()) # 转回Numpy是个列表