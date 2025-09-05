"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
  return MulScalar(scalar)(a)



class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs=node.inputs
        return out_grad*(rhs*power(lhs,add_scalar(rhs,-1))),out_grad*(power(lhs,rhs)*log(lhs))
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs=node.inputs[0]
        return out_grad*(self.scalar*power_scalar(lhs,self.scalar-1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a/b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs=node.inputs
        
        return divide(out_grad,rhs),divide(-multiply(out_grad,lhs),multiply(rhs,rhs))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        #print(numpy.transpose(a,self.axes).shape)
        new_axes=[i for i in range(len(a.shape))]
        if self.axes==None:
          new_axes[a.ndim-2]=a.ndim-1
          new_axes[a.ndim-1]=a.ndim-2
        else:
          new_axes[self.axes[0]]=self.axes[1]
          new_axes[self.axes[1]]=self.axes[0]
        return a.permute(new_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad,axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return  a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad,node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape=node.inputs[0].shape
        num_missing_dims=len(self.shape)-len(in_shape)
        in_shape_aligned = (1,) * num_missing_dims + in_shape
        sum_axes=[]
        for i in range(len(in_shape_aligned)):
            if in_shape_aligned[i]==1 and node.inputs[0].shape!=1:
                sum_axes.append(i)
        if(sum_axes):
            return summation(out_grad,axes=tuple(sum_axes)).reshape(in_shape)
        else:
            return out_grad.reshape(in_shape)
        raise NotImplementedError()
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            # multiple axes case
          for axis in reversed(sorted(self.axes)):
            a = a.sum(axis = axis)
          return a
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape=node.inputs[0].shape
        out_shape=list(in_shape)
        if self.axes is None :
            axes=[i for i in range(len(in_shape))]
        else:
          try:
            axes=list(self.axes)
          except TypeError:
            axes=[self.axes]
        for i in axes:
            out_shape[i]=1
        return broadcast_to(reshape(out_grad,tuple(out_shape)),in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs=node.inputs
        lgrad,rgrad=matmul(out_grad,transpose(rhs)),matmul(transpose(lhs),out_grad)
        if len(lgrad.shape)>len(lhs.shape):
            lgrad=summation(lgrad,axes=tuple(range(len(lgrad.shape)-len(lhs.shape))))
        if len(rgrad.shape)>len(rhs.shape):
            rgrad=summation(rgrad,axes=tuple(range(len(rgrad.shape)-len(rhs.shape))))

        return lgrad,rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad,node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data()
        return out_grad * Tensor(out > 0, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)




class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
      return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR 
      
      return 4*out_grad/(exp(node.inputs[0]*2)+2.0+exp(node.inputs[0]*(-2)))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n=len(args)
        new_shape=list(args[0].shape)
        new_shape.insert(self.axis,n)
        out=array_api.empty(new_shape,device=args[0].device)
        idxes=[slice(0,s) for s in new_shape]
        for i,arr in enumerate(args):
          idxes[self.axis]=slice(i,i+1)
          out[tuple(idxes)]=arr
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad,axis=self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n=A.shape[self.axis]
        new_shape=list(A.shape)
        new_shape.pop(self.axis)
        out=[]
        idxes=[slice(0,s) for s in A.shape]
        for i in range(n):
          idxes[self.axis]=slice(i,i+1)
          out.append(A[tuple(idxes)].compact().reshape(tuple(new_shape)))
        return tuple(out)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad,axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        a=a.compact()
        dil_factor=self.dilation+1
        new_shape=list(a.shape)
        new_stride=list(a._strides)
        for axis in self.axes:
          new_shape[axis]*=dil_factor
        arr=a.device.full(tuple(new_shape),0)
        slices=[slice(0,d) for d in a.shape]
        for axis in self.axes:
          if axis>a.ndim:
            continue
          slices[axis]=slice(0,arr.shape[axis],dil_factor)
        arr[tuple(slices)]=a
        return arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,axes=self.axes,dilation=self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        a=a.compact()
        dil_factor=self.dilation+1
        slices=[slice(0,d) for d in a.shape]
        for axis in self.axes:
          if axis>a.ndim:
            continue
          slices[axis]=slice(0,a.shape[axis],dil_factor)
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A=A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim=K*K*C_in
        new_H=(H-K+1)//self.stride
        new_W=(W-K+1)//self.stride

        tmp=A.as_strided(shape=(N,new_H,new_W,K,K,C_in),strides=(Ns,Hs*self.stride,Ws*self.stride,Hs,Ws,Cs))
        tmp=tmp.compact().reshape((N*new_H*new_W,inner_dim))
        out=tmp @ B.compact().reshape((inner_dim,C_out))
        return out.reshape((N,new_H,new_W,C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X,W=node.inputs
        N,H,w,C_in = X.shape
        K,_,_,C_out = W.shape
        X_grad=conv(dilate(out_grad,(1,2),self.stride-1),flip(W,(0,1)).transpose((2,3)),1,K-1-self.padding)
        W_grad=conv(X.transpose((0,3)),dilate(out_grad,(1,2),self.stride-1).transpose((0,1)).transpose((1,2)),stride=1,padding=self.padding).transpose((0,1)).transpose((1,2))
        return(X_grad,W_grad)
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


