import numpy as np
import operator
x1 = np.array([[4,56],
              [5,57],
              [6,58],
              [56,4],
              [57,5],
              [58,6]])
y1 = ['a','a','a','b','b','b']
x = np.array([43,24])
x_data = np.tile(x, (x1.shape[0],1))

diff = np.sqrt(np.sum((x_data-x1)**2, axis=1))
index = diff.argsort()
K = 5
result = dict()
for i in range(K):
    key = y1[index[i]]
    result[key] = result.get(key, 0) + 1
result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
print (result[0][0])
