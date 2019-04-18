"""
data generator(with and without augment)
"""
"""
原始的图片是vol.nrrd，格式是[depth,h,w]
标注的信息是label.nrrd,格式也是[depth,h,w]，每一个具体的点上的值就是该像素属于的类别
因为恒为512张,patch_depth 必须能够被512整除,同时考虑到UNet里面的Concat层，导致w必须是8的倍数才能，
所以要resize一些不合规的图片
"""
import numpy as np
import os
import nrrd
class DataGenerator(object):
    def __init__(self,img_dir,train_bs,val_bs,patch_depth=16,shape=None,
                 split_rate=0.1,augment=False,labels=5):
        self.img_dir=img_dir
        self.train_bs=1 if shape==None else train_bs
        self.val_bs=val_bs
        self.shape=shape
        self.patch_depth=patch_depth
        self.split_rate=split_rate
        self.augment=augment
        self.total_slices = 512 * len(os.listdir(self.img_dir))
        self.labels=labels

    def split(self):
        self.patch_number=int(self.total_slices/self.patch_depth)
        self.folder_patch_number=int(512/self.patch_depth)
        all_index=np.array(range(self.patch_number))
        val_len=int(self.split_rate*len(all_index))
        self.val_index=np.random.choice(all_index,size=val_len,replace=False)
        self.train_index=np.array([i for i in all_index if i not in self.val_index])
        self.steps_per_epoch=len(self.train_index)//self.train_bs
        self.vaild_steps=len(self.val_index)//self.val_bs
        print('val len: {}, train len: {}  patch number:{}'.format(len(self.val_index),len(self.train_index),
                                                                   len(self.val_index)+ len(self.train_index)))

    def generator(self,valid=False):
        index=self.train_index if valid==False else self.val_index
        batch_size=self.train_bs if valid==False else self.val_bs
        x=[]
        y=[]
        count=0
        if self.shape==None:
            if self.augment==False:
                while True:
                    np.random.shuffle(index)
                    for i in range(len(index)):
                        patch_id=index[i]
                        folder=os.listdir(self.img_dir)
                        folder_name=os.path.join(self.img_dir,folder[patch_id//self.folder_patch_number])
                        patch_index=patch_id%self.folder_patch_number
                        patch_x,_=nrrd.read(os.path.join(folder_name, 'CT-vol.nrrd'))
                        patch_x=np.expand_dims(patch_x,axis=-1)
                        #切割self depth厚度的图片
                        patch_x=patch_x[patch_index*self.patch_depth:(patch_index+1)*self.patch_depth,:,:,:]
                        """
                        normalization
                        """
                        patch_y,_=nrrd.read(os.path.join(folder_name, 'Segmentation-label.nrrd'))
                        patch_y=patch_y[patch_index*self.patch_depth:(patch_index+1)*self.patch_depth,:,:]
                        temp_s=patch_y.shape
                        patch_y=np.eye(self.labels)[patch_y.reshape(-1)]
                        patch_y=patch_y.reshape((temp_s[0],temp_s[1],temp_s[2],self.labels))
                        x.append(patch_x)
                        y.append(patch_y)
                        count+=1
                        if count>=batch_size:
                            x,y=np.array(x),np.array(y)
                            print(x.shape,y.shape)
                            yield x,y
                            count=0
                            x=[]
                            y=[]
            else:
                """
                数据增强函数
                """
                pass
        else:
            """
            resize(original map, seg map)
            """
            pass
if __name__=='__main__':
    d=DataGenerator('C:\\Users\chris.li2\\3D_medical',20,20)
    d.generator()