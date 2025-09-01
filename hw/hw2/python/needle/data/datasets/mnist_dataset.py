from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename,'rb') as im:
            
            
            magic,self.n,self.row,self.col=struct.unpack('>IIII',im.read(16))
            #print(n)
            self.X=np.frombuffer(im.read(self.n*self.row*self.col),dtype=np.uint8).astype(np.float32).reshape((self.n,self.row,self.col,1))/255.0
        
        with gzip.open(label_filename,'rb') as lb:
            
            
            magic,n=struct.unpack('>II',lb.read(8))
            #print(n)
            self.y=np.frombuffer(lb.read(n),dtype=np.uint8)
            ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X=self.apply_transforms(self.X[index])
        return (X,self.y[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.n
        ### END YOUR SOLUTION