"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for i, param in enumerate(self.params):
            ### BEGIN YOUR SOLUTION
            if param.grad is None:
                continue
            grad = (param.grad + self.weight_decay * param.data).detach()
            if i not in self.u:
                self.u[i] = (1 - self.momentum) * grad
            else:
                self.u[i] = (1 - self.momentum) * grad + self.momentum * self.u[i]
            param.data -= self.lr * self.u[i]
            ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0.0

        self.m = {}
        self.v = {}

    def step(self):
        self.t+=1.0
        for i, param in enumerate(self.params):
            ### BEGIN YOUR SOLUTION
            if param.grad is None:
                continue
            grad = (param.grad + self.weight_decay * param.data).detach()
            if i not in self.m:
                self.m[i] = (1 - self.beta1) * grad
            else:
                self.m[i] = (1 - self.beta1) * grad + self.beta1 * self.m[i]
            if i not in self.v:
                self.v[i] = (1 - self.beta2) * (grad**2)
            else:
                self.v[i] = (1 - self.beta2) * (grad**2) + self.beta2 * self.v[i]
            u=(self.m[i]/(1-self.beta1**self.t)).detach()
            v=(self.v[i]/(1-self.beta2**self.t)).detach()
            param.data -= self.lr * u/(v**0.5  +self.eps)

