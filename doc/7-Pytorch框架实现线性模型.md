今天我们继续来实现线性回归模型，不过这一次我们不再所有功能都自己实现，而是使用Pytorch框架来完成。



整个代码会发生多大变化呢？



首先是数据生成的部分，这个部分和之前类似：



```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
```



```python
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```



但是从数据读取开始，就变得不同了。



在之前的代码中，我们是自己实现了迭代器，从训练数据中随机抽取数据。但我们没有做无放回的采样设计，也没有做数据的打乱操作。



然而这些内容Pytorch框架都有现成的工具可以使用，我们不需要再自己实现了。



这里需要用到`TensorDataset`和`DataLoader`两个类：



```python
def load_array(data_arrays, batch_size, is_train=True): #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```



关于这两个类的用法，我们可以直接询问ChatGPT。




![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240130205806993.png)

简而言之`TensorDataset`是用来封装`tensor`数据的，它的主要功能就是和`DataLoader`配合。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240130205832583.png)

`DataLoader`是一个迭代器，除了基本的数据读取之外，还提供乱序、采样、多线程读取等功能。

我们调用`load_array`获得训练数据的迭代器。



```python
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```



## 模型部分



在之前的实现当中，我们是自己创建了两个`tensor`来作为线性回归模型的参数。



然而其实不必这么麻烦，我们可以把线性回归看做是单层的神经网络，在原理和效果上，它们都是完全一样的。因此我们可以通过调用对应的API来很方便地实现模型：



```python
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```



这里的`nn`是神经网络的英文缩写，`nn.Linear(2, 1)`定义了一个输入维度是2，输出维度是1的单层线性网络，等同于线性模型。



`nn.Sequential`模块容器，它能够将输入的多个网络结构按照顺序拼装成一个完整的模型。这是一种非常常用和方便地构建模型的方法，除了这种方法之外，还有其他的方法创建模型，我们在之后遇到的时候再详细展开。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240130212121856.png)



一般来说模型创建好了之后，并不需要特别去初始化，但如果你想要对模型的参数进行调整的话，可以使用`weight.data`和`weight.bias`来访问参数：



```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```



接着我们来定义损失函数，Pytorch当中同样封装了损失函数的实现，我们直接调用即可。



```python
loss = nn.MSELoss()
```



`nn.MSELoss`即均方差，MSE即mean square error的缩写。



最后是优化算法，Pytorch当中也封装了更新模型中参数的方法，我们不需要手动来使用`tensor`里的梯度去更新模型了。只需要定义优化方法，让优化方法自动完成即可：



```python
optim = torch.optim.SGD(net.parameters(), lr=0.03)
```



## 训练



最后就是把上述这些实现全部串联起来的模型训练了。


整个过程代码量很少，只有几行。



```python
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        optim.zero_grad()
        l.backward()
        optim.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```



我们之前自己实现的模型参数更新部分，被一行`optim.step()`代替了。



不论多么复杂的模型，都可以通过`optim.step()`来进行参数更新，非常方便！



同样我们可以来检查一下训练完成之后模型的参数值，同样和我们设置的非常接近。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240130230359334.png)



到这里，整个线性回归模型的实现就结束了。



这个模型是所有模型里最简单的了，正因为简单，所以最适合初学者。后面当接触了更多更复杂的模型之后，会发现虽然代码变复杂了，但遵循的仍然是现在这个框架。



最后，欢迎加入我的星球~



除了Ai什么都有，个人学习成长、理财投资、量化交易心得，持续更新中



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/%E6%98%9F%E7%90%83%E4%BC%98%E6%83%A0%E5%88%B8%20(6).jpeg)