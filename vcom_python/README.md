

# 问题及解决
## 相对引用出错
直接在vcom_python目录下执行`python util_ml.py`会提示找不到util_file包，需要在上级目录调用`python -m vcom_python.util_ml`可以解决问题，具体原因参照[链接](https://blog.csdn.net/junbujianwpl/article/details/79324814)。

> 解决模块的算法是基于__name__和__package__变量的值。大部分时候，这些变量不包含任何包信息 。比如：当直接使用python解释器执行脚本python scripty.py时，__name__= __main__ 和 __package__ = None ，python解释器不知道模块所属的包。在这种情况下，相对引用会认为这个模块就是顶级模块，而不管模块在文件系统上的实际位置。

# 没有chardet包
需要使用命令安装
```sh
conda install chardet
```
