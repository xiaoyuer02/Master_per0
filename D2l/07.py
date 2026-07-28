# ****************
# 加载图像分类数据集
# ****************

import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l

d2l.use_svg_display()

# 通过ToTensor是咧将图像数据从PIL类型变换为32点浮点数
# 并除以255使得所有像素数值从0到1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root = '../data', train = True, 
                                                transform = trans, download = False)
mnist_test = torchvision.datasets.FashionMNIST(root = '../data', train = False, 
                                                transform =  trans, download = False)

print('len(mnist_train): ', len(mnist_train))
print('len(mnist_test): ', len(mnist_test))
print('mnist_train[0][0].shape: ', mnist_train[0][0].shape)

def get_fashion_mnist_labels(labels):
    """ 返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 
                   'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    """ Plot a list of images """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize = figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.set_title(titles[i])
    d2l.plt.show()

X, y = next(iter(data.DataLoader(mnist_train, batch_size = 18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles = get_fashion_mnist_labels(y))

batch_size = 256

def get_dataloader_workers():
    """ 使用4个进程来读取的数据。 """
    return 4

train_iter = data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True,
                             num_workers = get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f"{timer.stop():f} sec") # 读取数据的时间性能要比训练时间快很多才行

def load_data_fashion_mnist(batch_size, resize = None):
    """ 下载Fashion-MNIST数据集， 然后将其加载到内存中 """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root = '../data', train = True, transform = trans,
                                                    download = False)
    mnist_test = torchvision.datasets.FashionMNIST(root = '../data', train = False, transform = trans, 
                                                    download = False)
    return (data.DataLoader(mnist_train, batch_size, shuffle = True, num_workers = get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle = True, num_workers = get_dataloader_workers()))