import nrrd
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
原始的图片是vol.nrrd，格式是h,w,深度
标注的信息是label.nrrd,格式也是H,w,深度，每一个具体的点上的值就是该像素属于的类别
"""
class data_loader(object):
    def __init__(self,data_dir):
        self.dir=data_dir
    def process(self):
        for i in os.listdir(self.dir):
            path=os.path.join(self.dir,i)
            for i in os.listdir(path):
                if i.endswith('vol.nrrd'):
                    filepath=os.path.join(path,i)
                    data,nrrd_options=nrrd.read(filepath)
                    print(i,data.shape)
                    # for j in nrrd_options:
                    #     print(j, nrrd_options[j])
                elif i.endswith('seg.nrrd'):
                    filepath = os.path.join(path, i)
                    data, nrrd_options = nrrd.read(filepath)
                    print(i,data.shape)
                    for j in nrrd_options:
                        print(j,nrrd_options[j])
                elif i.endswith('label.nrrd'):
                    filepath = os.path.join(path, i)
                    data, nrrd_options = nrrd.read(filepath)
                    print(i, data.shape)
                    data=np.reshape(data,(data.size,))
                    print(np.unique(data))
                    print('*'*90)
                    # for j in nrrd_options:
                    #     print(j, nrrd_options[j])

if __name__=='__main__':
    dir='C:\\Users\chris.li2\\3D_medical'
    d=data_loader(dir)
    d.process()

