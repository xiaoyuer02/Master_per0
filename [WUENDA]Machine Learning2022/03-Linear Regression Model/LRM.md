# Task-利用已知数据，为线性回归模型找到拟合参数$\hat{w}$ $b$，最小化二乘项代价函数$cost-J$
## 成本（代价）函数J
正则化平均平方误差Regularized MSE-Mean Squared Error

正则化项用于减少模型特征复杂度带来的过拟合影响
$$J = \frac{1}{2m}\sum_{i=0}^{m-1}{(f_{\hat{w},b}(\hat{x_{i}})-y_{i})^{2}}+\frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$$
> 标准做法不考虑对b正则化