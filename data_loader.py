import nrrd
from PIL import Image
import numpy as np
import os
import time
"""
原始的图片是vol.nrrd，格式是h,w,深度
标注的信息是label.nrrd,格式也是h,w,深度，每一个具体的点上的值就是该像素属于的类别
"""
class data_loader(object):
    def __init__(self,data_dir):
        self.dir=data_dir
    def process(self,mode=2):
        for i in os.listdir(self.dir):
            path=os.path.join(self.dir,i)
            for i in os.listdir(path):
                if mode==1:
                    if i.endswith('vol.nrrd'):
                        filepath = os.path.join(path, i)
                        data, nrrd_options = nrrd.read(filepath)
                        print(i, data.shape)
                        print(np.unique(data))
                        # for j in nrrd_options:
                        #     print(j, nrrd_options[j])
                    elif i.endswith('seg.nrrd'):
                        filepath = os.path.join(path, i)
                        data, nrrd_options = nrrd.read(filepath)
                        print(i, data.shape)
                        for j in nrrd_options:
                            print(j, nrrd_options[j])
                    elif i.endswith('label.nrrd'):
                        filepath = os.path.join(path, i)
                        data, nrrd_options = nrrd.read(filepath)
                        print(i, data.shape)
                        data = np.reshape(data, (data.size,))
                        print(np.unique(data))
                        print('*' * 90)
                        # for j in nrrd_options:
                        #     print(j, nrrd_options[j])
                else:
                    """
                    用不同颜色显示实例分割情况,得出的结论是：
                    label是这样的格式：[depth,h,w]
                    """
                    original_pictures,nrrd_options=nrrd.read(os.path.join(path,'CT-vol.nrrd'))
                    labels,_=nrrd.read(os.path.join(path,'Segmentation-label.nrrd'))
                    map={
                        0:[0,0,0],
                        1:[255,0,0],
                        2:[0,255,0],
                        3:[0,0,255],
                        4:[255,255,0]
                    }
                    print(original_pictures.shape,labels.shape)
                    # print(original_pictures[0,0,0])
                    # original_pictures=np.expand_dims(original_pictures,axis=-1)
                    # print(original_pictures.shape)
                    # t=original_pictures[0,0,0,:]
                    # print(t)
                    # i=0
                    # print(original_pictures[i, :, :].shape)
                    # single_img = Image.fromarray(original_pictures[i, :, :])
                    # single_img.show()
                    # w, h = single_img.size
                    # print(np.unique(labels[i, :, :]))
                    # mask = Image.new('RGB', size=(w, h))
                    # mask = np.array(mask)
                    # for x in range(h):
                    #     for y in range(w):
                    #         mask[x, y, :] = map[labels[i, x, y]]
                    # mask_img = Image.fromarray(mask)
                    # mask_img.show()
                    # time.sleep(3)
                    for i in range(original_pictures.shape[0]):
                        shape=original_pictures[i,:,:].shape
                        print(shape)
                        mask = Image.new('RGB', size=(shape[1], shape[0]))
                        mask = np.array(mask)
                        flag=False
                        for x in range(shape[0]):
                            for y in range(shape[1]):
                                mask[x, y, :] = map[labels[i, x, y]]
                                if labels[i, x, y] in [1,3,4]:
                                    flag=True
                        if flag:
                            mask_img = Image.fromarray(mask)
                            mask_img.show()
                            time.sleep(3)
if __name__=='__main__':
    dir='C:\\Users\chris.li2\\3D_medical'
    d=data_loader(dir)
    d.process()

