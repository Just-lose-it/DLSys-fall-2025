import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
from needle.data import DataLoader
from needle.data.datasets import MNISTDataset
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Residual(nn.Sequential(nn.Linear(dim,hidden_dim),norm(hidden_dim),nn.ReLU(),nn.Dropout(drop_prob),nn.Linear(hidden_dim,dim),norm(dim))),nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(nn.Linear(dim,hidden_dim),nn.ReLU(),*[ResidualBlock(hidden_dim,hidden_dim//2,norm,drop_prob) for _ in range(num_blocks)],nn.Linear(hidden_dim,num_classes))
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    err_num=0.0
    loss_sum=[]
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
        for X,y in dataloader:
            print('-',end='')
            opt.reset_grad()
            X=X.reshape((X.shape[0],784))
            #print(X.shape)
            logits=model.forward(X)
            err_num+=np.sum(logits.numpy().argmax(axis=1)!=y.numpy())
            loss=nn.SoftmaxLoss().forward(logits,y)
            loss_sum.append(loss.numpy())
            loss.backward()
            opt.step()
        
    else:
        model.eval()
        for X,y in dataloader:
            print('-',end='')
            X=X.reshape((X.shape[0],784))
            logits=model(X)
            err_num+=np.sum(logits.numpy().argmax(axis=1)!=y.numpy())
            loss=nn.SoftmaxLoss().forward(logits,y)
            loss_sum.append(loss.numpy())
    print()
    return err_num/len(dataloader.dataset),np.mean(loss_sum)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tforms = [ndl.data.RandomCrop(15), ndl.data.RandomFlipHorizontal()]
    model=MLPResNet(784,hidden_dim,3,10)
    opt=optimizer(params=model.parameters(),lr=lr,weight_decay=weight_decay)
    train_dataset=MNISTDataset(f'{data_dir}/train-images-idx3-ubyte.gz',f'{data_dir}/train-labels-idx1-ubyte.gz')
    test_dataset=MNISTDataset(f'{data_dir}/t10k-images-idx3-ubyte.gz',f'{data_dir}/t10k-labels-idx1-ubyte.gz')
    test_loader=DataLoader(test_dataset,batch_size)
    train_loader=DataLoader(train_dataset,batch_size,shuffle=True)
    for i in range(epochs):
        print("Epoch "+str(i+1)+' / '+str(epochs))
        train_error,train_loss=epoch(train_loader,model,opt)
    test_error,test_loss=epoch(test_loader,model)
    ### END YOUR SOLUTION
    return (train_error,train_loss,test_error,test_loss)


if __name__ == "__main__":
    print(train_mnist(data_dir="../data"))

