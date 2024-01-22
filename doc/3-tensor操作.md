这一章节我们来将`tensor`的运算和操作。



## 运算



我们在介绍`tensor`的时候说过，它本质上就是一个多维矩阵，但是基于深度神经网络模型的需求以及性能的要求，封装了很多功能，以及一些性能优化。



比如在原生Python当中，我们是不能对一个`list`实例直接做运算的，只能遍历`list`内的元素，对其做运算。而在深度学习当中，一个`tensor`的维度可能很多，内部包含的元素数量更多，动辄数百万甚至上千万，我们肯定是不可能使用循环去做计算的，这太慢了。



所以和Python原生的`list`等对象相比，第一点巨大的不同，`tensor`能够整体执行运算。



先从最基础的四则运算开始：



```python
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([2, 3, 1, 2])

x + y, x - y, x * y, x / y
```



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240122213041217.png)



从结果当中我们可以看到，对于两个`shape`相同的`tensor`执行四则运算，等价于它们对应位置的元素两两运算。



乘方运算也是支持的，和Python中一样，使用`**`：



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240122213249596.png)



我们可以直接使用Python中原生的运算符，是因为`tensor`内部重载了这些实现。如果不嫌麻烦，也可以使用Pytorch中提供的api，实现的效果是一样的。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240122213501930.png)



除了基础的四则运算之外，还有另外两个比较常用的，一个是`tensor.exp()`计算元素的指数，一个是`tensor.sqrt()`计算元素的平方根。



比较值得说的是矩阵乘法`tensor.matmul`，它和numpy中的`.dot`类似，计算矩阵乘法。



比如一个shape为`(3, 2)`的矩阵乘上一个shape是`(2, 3)`的矩阵，得到的结果是一个`(3, 3)`的矩阵。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240122215144132.png)



`tensor`当中和计算相关的api基本上就是这些，我个人感觉还是挺直观的。



这还没有结束，除了运算api，`tensor`当中还有一些其他的变换操作也非常常用。和运算针对的是两个或多个`tensor`不同，操作针对的是单个`tensor`，下面我们结合例子来看一下。



## 操作



我们在上篇文章当中介绍了`shape`的概念，既然`tensor`的本质是一个多维矩阵，如果我们想要改变一个`tensor`的`shape`有没有办法呢？



当然是有的，这会涉及到两个api：`view`和`reshape`。



我们先来看下效果：



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240122220055918.png)



从效果上来看，这两种方式的效果是一样的。既然如此，那为什么要创造出两个不同名的api呢？



老实讲我之前读过的基本Pytorch资料当中都没有提及，我也一直没有留意这个细节，直到这一次写文章才注意到。带着这个问题我去询问了ChatGPT，它的回答如下：



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240122220334100.png)



简单来说，`view`要求元素的内存连续，否则会报错，好处是效率更高。而`reshape`更加兼容，可以适应几乎所有情况，但性能稍低。



这里有个小疑问：`tensor`难道不是连续储存的吗？为什么会出现不连续的情况呢？这个问题我之前也没想过，好在有ChatGPT，做出了比较靠谱的回答。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240122234958563.png)



`tensor`在初始化之后的确是连续存储的，只不过在进行矩阵转置操作之后，可能会引起不连续。



如果你还想知道为什么转置会引发元素的不连续，还可以继续追问。



![](https://moutsea-blog.oss-cn-hangzhou.aliyuncs.com/image-20240122235340376.png)



总之，在除去执行了转置的情况之外，其余情况`view`和`reshape`可以通用。



