![logo.png](logo.png)

MagicFlow是在PyTorch与Paddle之上的一层封装，目的是为了用同一套代码在不同的深度学习框架上训练模型。

:warning: 本项目还处于概念验证阶段，目前只支持一部分功能。

## 简要用法

手动自定义深度学习引擎，在使用mf之前通过调用```mf.config('torch'或'paddle'')```指定使用的深度学习框架。

```python
import mf

mf.config(framework='torch')  # noqa
from mf import nn
```

或者可通过如下代码自动选择引擎（如果安装PyTorch会被优先选择）：

```python
import mf.auto
from mf import nn
```

在使用深度学习组件时，无需指定输入的通道数，输入的通道数由第一个输入到模型的Tensor确定。

```python
nn.Sequential(
    nn.Conv2D(64, kernel_size=3),
    nn.BatchNorm2D(),
    nn.ReLU()
)
```
