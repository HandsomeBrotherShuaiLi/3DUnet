import nrrd
from PIL import Image
import numpy as np
import os
import time
"""
原始的图片是vol.nrrd，格式是[h,w depth,]
标注的信息是label.nrrd,格式也是[h,w depth,]，每一个具体的点上的值就是该像素属于的类别
"""
class data_loader(object):
    def __init__(self,data_dir):
        self.dir=data_dir
    def process(self,mode=2):
        for i in os.listdir(self.dir):
            path=os.path.join(self.dir,i)
            """
                               用不同颜色显示实例分割情况,得出的结论是：
                               label是这样的格式：[h,w,depth]
                               """
            # original_pictures, nrrd_options = nrrd.read(os.path.join(path, 'CT-vol.nrrd'))
            labels, _ = nrrd.read(os.path.join(path, 'Segmentation-label.nrrd'))
            # for j in _:
            #     print(j,_[j])
            # map = {
            #     0: [0, 0, 0],
            #     1: [255, 0, 0],
            #     2: [0, 255, 255],
            #     3: [0, 0, 255],
            #     4: [255, 255, 0]
            # }
            print('before',np.unique(labels))
            labels[np.where(labels==3)]=0
            labels[np.where(labels==4)]=3
            t1,t2,t3,t4=np.where(labels==0),np.where(labels==1),np.where(labels==2),np.where(labels==3)
            print(labels[t1].size,labels[t2].size,labels[t3].size,labels[t4].size,(labels[t1].size+labels[t2].size+labels[t3].size+labels[t4].size)==labels.size)
if __name__=='__main__':
    dir='C:\\Users\chris.li2\\3D_medical'
    d=data_loader(dir)
    d.process()

