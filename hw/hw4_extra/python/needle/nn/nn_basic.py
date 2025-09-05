"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import needle as ndl
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x



class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.b=bias
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.kaiming_uniform(fan_in=in_features,fan_out=out_features,requires_grad=True,device=device))#in,out
        if bias:
          self.bias=Parameter(init.kaiming_uniform(out_features,1,requires_grad=True,device=device).transpose())#1,out
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N=X.shape[0]
        if self.b:
            return X@self.weight + self.bias.broadcast_to((N,self.out_features))
        else:
            return X@self.weight 
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X:Tensor):
        ### BEGIN YOUR SOLUTION
        tot_dims=1
        for i in range(1,len(X.shape)):
            tot_dims *=X.shape[i]
        return X.reshape((X.shape[0],tot_dims))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ndl.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        cur=x
        for module in self.modules:
            cur=module(cur)
        return cur
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        #print(logits.shape,y.shape,init.one_hot(logits.shape[1],y).shape,ndl.logsumexp(logits,axes=(1,)).shape)
        z_y=logits*init.one_hot(logits.shape[1],y,device=y.device)
        return (ndl.logsumexp(logits,axes=(1,))-z_y.sum(axes=(1,))).sum()/logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones(dim,device=device,dtype=dtype,requires_grad=True))
        self.bias=Parameter(init.zeros(dim,device=device,dtype=dtype,requires_grad=True))
        self.running_means=init.zeros(dim,device=device,dtype=dtype)
        self.running_var=init.ones(dim,device=device,dtype=dtype)
        ### END YOUR SOLUTION
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        
        
        if(self.training):
            ex=x.sum(axes=(0,))/x.shape[0]
            exs=ex.reshape((1,x.shape[1])).broadcast_to(x.shape)
            varx=ndl.power_scalar(x-exs,2.0).sum(axes=(0,))/x.shape[0]
            self.running_means=((1-self.momentum)*self.running_means+self.momentum*ex.cached_data)
            self.running_var=((1-self.momentum)*self.running_var+self.momentum*varx.cached_data)
        else:
            exs=self.running_means.broadcast_to(x.shape)
            varx=self.running_var
        return self.weight.reshape((1,x.shape[1])).broadcast_to(x.shape)*(x-exs)/(ndl.power_scalar(varx.reshape((1,x.shape[1])).broadcast_to(x.shape)+self.eps,0.5))+self.bias.reshape((1,x.shape[1])).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight=Parameter(init.ones(1,dim,device=device,dtype=dtype,requires_grad=True))
        self.bias=Parameter(init.zeros(1,dim,device=device,dtype=dtype,requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        weight=self.weight.broadcast_to(x.shape)
        bias=self.bias.broadcast_to(x.shape)
        ex=x.sum(axes=(1,)).reshape((x.shape[0],1))/x.shape[1]
        exs=ex.broadcast_to(x.shape)
        varx=ndl.power_scalar(x-exs,2.0).sum(axes=(1,)).reshape((x.shape[0],1)).broadcast_to(x.shape)/x.shape[1]
        return weight*(x-exs)/(ndl.power_scalar(varx+self.eps,0.5))+bias
        ### END YOUR SOLUTION

#P.S. (Added by Just-lose-it)The test will use given np random seed and will require the same result,
#So "prob 1-p" case corresponds to "mask < 1-self.p", not "mask>self.p"(though equal in practical cases) 
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = (np.random.rand(*x.shape) < 1-self.p) / (1 - self.p)
            
            mask_tensor=ndl.Tensor(mask,device=x.device,dtype=x.dtype)
            return x*mask_tensor
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION
