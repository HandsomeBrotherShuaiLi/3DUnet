import nrrd
from PIL import Image
import numpy as np
import os

class data_loader(object):
    def __init__(self,data_dir):
        self.dir=data_dir
    def process(self):
        for i in os.listdir(self.dir):
            path=os.path.join(self.dir,i)
            for i in os.listdir(path):
                if i.endswith('nrrd'):
                    filepath=os.path.join(path,i)
                    data,nrrd_options=nrrd.read(filepath)
                    print(i,data.shape)
if __name__=='__main__':
    d=data_loader('D:\\to-lishuai')
    d.process()
