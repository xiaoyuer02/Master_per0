# *******
# 线性代数
# *******
import torch
x = torch.tensor([3.0]) # 标量
y = torch.tensor([2.0])
print(x, y, x + y, x - y, x * y, x / y)

x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

A = torch.arange(20).reshape(5, 4) # 矩阵
print(A,'\n', A.T)
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T) # 对称矩阵转置后不变

X = torch.arange(24).reshape(2, 3, 4)
print(X)

A = torch.arange(20).reshape(5, 4)
B = A.clone()  # 分配新内存 A副本给B
print(A, '\n', A+B, id(A) == id(B))

print(A*B) # 哈达玛积

a = 2 # 按元素作二元运算
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X).shape, a*[2, 3, 4]) # 列表里面*是重复 要转到numpy做数学乘法

x = torch.arange(4)
print(x, x.sum())

A = torch.arange(20*2).reshape(2, 5, 4)
print(A, A.sum())
A_sum_0 = A.sum(axis = 0)
print(A_sum_0, A_sum_0.shape)
A_sum_1 = A.sum(axis = 1)
print(A_sum_1, A_sum_1.shape)
A_sum_0_1 = A.sum(axis = [0, 1])
print(A_sum_0_1, A_sum_0_1.shape)

print(A.float().mean(), A.float().mean(axis = 0))
print(A.sum() / A.numel(), A.sum(axis = 0) / A.shape[0])

sum_A = A.sum(axis = 0, keepdim = True) # 保留维数，为1
print(sum_A) 
print(A/sum_A) # 后续就可以利用广播机制进行计算，否则报错
print(A.cumsum(axis = 0)) # 某个轴上的元素累加求和

y = torch.ones(4)
print(x, y, torch.dot(x.float(), y)) # 向量间点积
print(x*y, (x*y).type())
print(torch.sum(x*y)) # 直接乘法再求和实现相同功能点积

A = torch.arange(20).reshape(5, 4)
x = torch.arange(4, dtype = torch.float32)
print(A.float().type(), x.type(), torch.mv(A.float(), x)) # 矩阵向量积
B = torch.ones(4, 3, dtype = torch.float32)
print(torch.mm(A.float(), B)) # 矩阵矩阵乘法
# .float()为float32, dtype = float为float64 数据类型不一致

u = torch.tensor([3.0, -4.0])
# 对向量
print(torch.norm(u)) # L2范数 元素平方后求和再开方
print(torch.abs(u).sum()) # L1范数 元素绝对值后求和
# 对矩阵
print(torch.norm(torch.ones(3, 3))) # 弗罗贝尼乌斯范数 矩阵元素平方和的平方根

