
# coding: utf-8

# In[6]:


#!/usr/bin/env python3


# In[7]:


import random
import numpy as np


# In[8]:


class Data:
    """
    Data Encapsulate.
    
    1. Attribute
    2. Targets
    """
    def __init__(self, attr, target = None):
        self.attr = attr
        self.target = target
    def getAttr(self):
        return self.attr
    def getTarget(self):
        return self.target
    def __str__(self):
        return 'Attrs: ' + str(self.attr) + ', Targets: ' + str(self.target)


# In[9]:


class DataContainer:
    """
    Contain Datas, and providing some method to return training batch.
    """
    def __init__(self, datas = None):
        """
        Arg:
        --------------------------
            * datas <List of type 'Data'> 
        """
        if datas is not None:
            ## Check Data type
            if type(datas[0]) is not Data:
                raise TypeError('DataContainer can only store "Data" type!')
            self.datas = datas
            self.toGenerate = random.sample(datas, len(datas))
        else:
            self.datas = []
            self.toGenerate = []
    
    
    def getAttrs(self):
        """
        Return all attributes for each data.
        """
        attrs = [data.getAttr() for data in self.datas]
        return np.array(attrs)
    
    def getTargets(self):
        """
        Return all targets for each data.
        """
        targets = [data.getTarget() for data in self.datas]
        return np.array(targets)
    
    def nextBatch(self, batch_size):
        """
        Return a batch_size DataContainer contains datas for a batch.
        """
        ## Make sure we have enough data to generate
        if len(self.toGenerate) < batch_size:
            self.toGenerate = random.sample(self.datas, len(self.datas))
        ## Pop out batch size number of data
        batch = self.toGenerate[:batch_size]
        ## Remove those data from list
        self.toGenerate = self.toGenerate[batch_size:]
        return DataContainer(batch)
    
    def nextEpoch(self, batch_size):
        """
        Return batches in one epoch. 
        """
        numBatches = len(self) // batch_size
        toGenerate = random.sample(self.datas, len(self.datas))
        for i in range(numBatches):
            batch = toGenerate[:batch_size]
            toGenerate = toGenerate[batch_size:]
            yield DataContainer(batch)
    
    def merge(self, other):
        self.datas = self.datas + other.datas
#         self.toGenerate = random.sample(self.datas, len(self.datas))
        
    def getAllDatas(self):
        for d in self.datas:
            yield d
            
    def __str__(self):
        output = ['* ' + d.__str__() for d in self.datas]
        return '\n'.join(output)
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, key):
        return self.datas[key]
        


# In[16]:


if __name__ == '__main__':
    c = []
    b = []
    for i in range(5):
        c.append(Data(list(range(i + 1)), i))
        b.append(Data(list(range(10 - i)), 10 - i))
    c = DataContainer(c)
    b = DataContainer(b)
    a = DataContainer()
    a.merge(b)
    a.merge(c)
    
    for batch in a.nextEpoch(batch_size = 5):
        print(batch)
        print()
    
    for batch in a.nextEpoch(batch_size=3):
        print(batch)
        print()
        
    

