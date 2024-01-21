这一章节主要介绍`Tensor`的概念。



## tensor的概念



在深度学习当中，`tensor`是一个非常重要的核心概念。作为非英文母语者，在接触的时候可能会有一些困惑和陌生，无形中这也会增加学习的成本。



比如有些同学可能会觉得tensor和谷歌的TensorFlow框架是否有什么关联，其实并没有。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240121170835434.png)



我们来看下ChatGPT对tensor的解释。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240121171359369.png)



ChatGPT的解释非常到位，只不过对于初学者来说可能很难体会到，所以我给大家标出了重点。



我提取一下大意，简单来说，tensor可以视作是多维数组，作用是存储数据。然后基于tensor我们可以执行一些运算，还能使用GPU进行加速。



也就是说在深度学习当中，框架（Pytorch、TensorFlow等）直接操作的对象并不是Python中的原生对象（list、tuple），而是tensor。可以理解成数据都要被转化成tensor的形式才能被模型使用。



可能大家会觉得疑惑，为什么不使用Python原生的数据结构而要单独创造出这么一个新概念呢？这不是增加了学习和使用的成本吗？



这里面原因很多，但是说穿了核心原因只有一个，就是Python的执行效率太低了。



程序员们普遍认为Python比较好用，当初Python成为机器学习领域的主力语言的核心原因就是如此。因为语法简单易用易学，使得大量学术圈的研究者们选择使用Python来进行模型的实验以及相关研究工作。通常高度抽象封装的语言会比较易用，但也因此往往会牺牲性能。Python作为动态类型的解释型语言，性能和C++这样的编译型语言有着巨大的鸿沟。



所以一些计算密集型的工具包底层都是通过C++或者Java实现的，Python只是上层封装了一个调用接口。常见的深度学习的计算框架几乎清一色都是C++实现的，另外就是运算的过程当中需要用到GPU来加速，Python原生类型并不支持。



不仅是Python原生类型不支持GPU加速，即使是numpy这样的科学计算包也不支持GPU。



我这里只是简单介绍，大家如果感兴趣可以去询问ChatGPT获取更多细节。不知道大家有没有注意到GPU在其中起到的作用。可能很多人不知道，使用GPU对模型训练进行加速的技术对于AI行业来说，称作是力挽狂澜也不为过。像是神经网络、反向传播等算法其实早在几十年前就已经被提出了，只不过那时候限制于算力，人们无法确定这样的方法能不能收敛并达到一个比较好的效果。



GPU加速的出现，解开了这个关键的困局。



## 创建tensor



这一章剩余的篇幅我们来聊聊`tensor`。



首先从创建`tensor`开始，Pytorch当中创建`tensor`的方法有很多，难度不大，但是有一些细节需要讲清楚。



我们先来看最简单的构造函数：



```python
import torch

torch.tensor(10)
```



我们传入了一个数字10，调用了构造函数`torch.tensor`创建了一个`tensor`。



我们在jupyter里运行，可以直接看到结果。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240121174029695.png)



我们稍作修改，传入一个`list`，而不是`int`，会得到什么呢？



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240121174159495.png)



依然是一个`tensor`，不过这个`tensor`里多了一个`[]`。那么它和上面那个`tensor`是一样的吗？有什么区别吗？



```python
a = torch.tensor(10)
b = torch.tensor([10])
```



当然有区别，为了讲清楚区别， 我们需要介绍关于`tensor`的另外一个概念：`shape`。



### shape



我们直接询问ChatGPT，不仅能给出解释，还有示例。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240121174706643.png)



前面说了，`tensor`的本质是一个多维的数组，那么`shape`就是它的维度。我们可以直接通过`.shape`来获取一个`tensor`的维度。基于此，我们来看看刚刚那两个`tensor`的`shape`一不一样。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240121174830515.png)



果然是不一样的，`a`的`shape`是空的`list`，而`b`的不为空。`shape`为空表示这不是一个向量，而是标量，它们虽然包含的数值相同，但是意义是不同的，在**实际运算的时候是不能混用的！**



`shape`可以说是`tensor`最重要的属性，我们在使用时一定要搞清楚每个变量的`shape`，这是算法工程师的基本功。



我们继续来看`tensor`的创建，除了通过`torch.tensor`方法创建`tensor`之外，还有其他几种常用的方式。比如通过numpy创建。



numpy也是机器学习领域非常常用的科学计算库，它提供了一系列矩阵运算的api，我们也可以从numpy的array来创建`tensor`。如果你对numpy不熟悉也没有关系，可以先做大概了解，后面用到了再学习细节。



```python
import numpy as np

arr = np.array(10)
torch.from_numpy(arr)
```



我们在jupyter里运行会发现，它输出的除了10，还多了一个`dtype=torch.int32`。这多出来的内容，是另外一个`tensor`的关键属性——类型。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240121175524440.png)



### dtype



`dtype`很好理解，就是类型，和Python中的类型类似。Pytorch中支持的类型如下：

- `torch.float32`或`torch.float` —— 32位浮点数
- `torch.float64`或`torch.double` —— 64位双精度浮点数
- `torch.float16`或`torch.half` —— 16位半精度浮点数
- `torch.int8` —— 带符号8位整数
- `torch.uint8` —— 无符号8位整数
- `torch.int16`或`torch.short` —— 带符号16位整数
- `torch.int32`或`torch.int` —— 带符号32位整数
- `torch.int64`或`torch.long` —— 带符号64位整数



我们可以使用`.dtype`来获取一个`tensor`的类型。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240121180348915.png)



刚才的两个`tensor`并不是没有类型，只不过因为是默认类型，所以隐掉了。



我们可以对`tensor`使用`.long()`或者`.float()`来改变类型，也可以使用`to`方法：



```python
a.long()
b.double()

a = a.to(torch.int32)
b = b.to(torch.float)
```



理解了`shape`和`dtype`两个概念之后，我们再来看其他几种常用的`tensor`的构造方法：



```python
torch.ones(size)  # 返回shape=size，全为1的tensor
torch.zeros(size) # 返回shape=size，全为0的tensor
torch.rand(size)  # 返回shape=size，随机值的tensor
torch.randn(size) # 返回shape=size，标准正态分布随机值的tensor
```



注意这几种构造函数返回的`tensor`默认类型都是`torch.float`