# ********
# 数据预处理
# ********
import os
import torch # torch在后续才import会dll报错 放在最开头

os.makedirs(os.path.join('..', 'data'), exist_ok = True)
data_file = os.path.join('..', 'data', 'data.csv')
with open(data_file, 'w') as f:
    f.write('NumRoom, Alley, Price\n')
    f.write('NA,Pave,100000\n')
    f.write('2,NA,120000\n') # NA前面有空格会被识别为字符串 得删掉空格    
    f.write('4,NA,140000\n') # woc
    f.write('NA,NA,160000\n') 

import pandas as pd

data = pd.read_csv(data_file)
print(f'data = \n{data}')
input, output = data.iloc[:, 0:2], data.iloc[:, 2]
print(f'input = \n{input}, \noutput = \n{output}')

input = input.fillna(input.mean(numeric_only = True)) # 对于字符串类型的缺失值无法计算均值，所以要加上numeric_only = True
print(f'input_new: \n{input}')
# dummy是指将分类变量转换为虚拟变量（dummy variable），也就是one-hot编码
input = pd.get_dummies(input, dtype = int, dummy_na = True) # dummy_na = True表示将缺失值也作为一个类别
# dtype = int表示将虚拟变量的类型设置为整数类型 否则显示bool类型
print(f'input_new_new: \n{input}')


x = torch.tensor(input.values) # 转成张量 默认python里面float64 但一般深度学习选择float32
y = torch.tensor(output.values)
print(f'x = {x}')
print(f'y = {y}')


