# 函数拟合报告

## 函数定义

我们考虑拟合的目标函数为：

$f(x) = x^2 + x \cdot \cos(x) + \sin(x)$

```python
def target_function(x):
    return x**2 + x * np.cos(x) + np.sin(x)
```

## 数据采集

我们在区间 [-2π, 2π] 上均匀采样1000个数据点作为训练数据。

```python
x_train = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
y_train = target_function(x_train)
```

## 模型描述

我们使用了一个两层的前馈神经网络来拟合目标函数。该网络包含两个隐藏层，每个隐藏层包含32个神经元，并使用ReLU激活函数。输出层为一个神经元，不使用激活函数。

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

## 拟合效果

经过100个epoch的训练，模型取得了较好的拟合效果。下图展示了原始函数与模型预测函数之间的比较：

![1710249591285](image/function_fitting/1710249591285.png)

从图中可以看出，模型成功拟合了原始函数，并且在大部分区域表现良好.
