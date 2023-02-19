import sys
sys.path.append('./python')
import needle as ndl
import numpy as np
da=ndl.data.CIFAR10Dataset("data/cifar-10-batches-py",True)
print(da.X.shape)
print(da.y.shape)