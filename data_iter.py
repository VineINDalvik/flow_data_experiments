import numpy as np

X = np.array([[[1,2],[1,2]],[[3,4],[3,4]]])
y = np.array([[1],[2]])
num = 2

import random
def data_iter(batch_size):
    idx = list(range(num))
    random.shuffle(idx)
    for batch_i, i in enumerate(range(0,num,batch_size)):
        j = np.array(idx[i:min(i+batch_size, num)])
        yield batch_i, np.take(X,j,axis=0), np.take(y,j, axis=0)

for b, i, j in data_iter(1):
    print b
    print i,j

