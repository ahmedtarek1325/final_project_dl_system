import numpy as np 
import sys 

sys.path.append("./python")

import needle as ndl
from needle import backend_ndarray as nd
import needle.nn as nn 

########################
#                       #  
#   bmm                 #
#                       #
#########################
'''x = np.random.rand(4,5,3)
y = np.random.rand(4,3,7)
z = x @ y 
print("after numpy the shape is",z.shape)

A = ndl.Tensor(nd.array(x), device=ndl.cpu())
B = ndl.Tensor(nd.array(y), device=ndl.cpu())
print("A",type(A))
z2 = ndl.bmm(A,B)
print("after numpy the shape is",z2.shape)
print(np.linalg.norm(z2.numpy()-z))'''

'''########################
#                       #  
#   new splitter        #
#                       #
#########################
x = np.random.rand(4,9,5,3)

A = ndl.Tensor(nd.array(x), device=ndl.cpu())
print("A",type(A))
L_ = ndl.Split_group(axis = 1,splits= 3)(A)
y =np.split(x,3,1)

for i in range(3):
    print(y[i].shape)
    print(np.linalg.norm(L_[i].numpy()-y[i]))


y =np.split(x,1,3)
print(np.linalg.norm(L_-y))'''
'''
########################
#                       #  
#   new concatenator    #
#                       #
#########################

list_ = []
for i in range(3):
    list_.append(np.random.rand(4,9,5,3))

n_l = []
for l in list_:
    n_l.append(ndl.Tensor(nd.array(l), device=ndl.cpu()))

L_ = ndl.concat(n_l,2)

l2 = np_conca = np.concatenate(list_,axis = 2)

print(np.linalg.norm(L_.numpy()-l2))
'''



