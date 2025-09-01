from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def __init__(self):
        self.axes = (1,)
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z=array_api.max(Z,axis=self.axes,keepdims=True)
        
        return Z-array_api.log(array_api.sum(array_api.exp(Z-max_Z),axis=self.axes,keepdims=True))-max_Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z=node.inputs[0]
        
        return out_grad-summation(out_grad,axes=(1,)).reshape((out_grad.shape[0],1)).broadcast_to(Z.shape)*(logsoftmax(Z).exp())
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z=array_api.max(Z,axis=self.axes,keepdims=True)
        max_Z2=array_api.max(Z,axis=self.axes,keepdims=False)
        return array_api.log(array_api.sum(array_api.exp(Z-max_Z),axis=self.axes))+max_Z2
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        
        in_shape=node.inputs[0].shape
        out_shape=list(in_shape)
        if self.axes is None :
            axes=[i for i in range(len(in_shape))]
        else:
            axes=self.axes
        for i in axes:
            out_shape[i]=1
        Z=node.inputs[0]
        #print(Z.shape)
        
        return broadcast_to(reshape(out_grad,tuple(out_shape)),in_shape)*exp(Z-broadcast_to(logsumexp(Z,axes=self.axes).reshape(tuple(out_shape)),in_shape))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

