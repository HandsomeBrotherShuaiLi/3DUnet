"""
data generator(with and without augment)
"""
"""
原始的图片是vol.nrrd，格式是[depth,h,w]
标注的信息是label.nrrd,格式也是[depth,h,w]，每一个具体的点上的值就是该像素属于的类别
因为恒为512张,patch_depth 必须能够被512整除
发现 1*8*512*512的内存不够，1*4*512*512 pooling operation 不行
考虑修改 shape
"""
import numpy as np
import os
import nrrd
class DataGenerator(object):
    def __init__(self,img_dir,train_bs=4,val_bs=1,patch_depth=8,factor=4,shape=None,
                 split_rate=0.1,augment=False,labels=5):
        self.img_dir=img_dir
        self.train_bs=train_bs
        self.val_bs=val_bs
        self.shape=shape
        self.patch_depth=patch_depth
        self.split_rate=split_rate
        self.augment=augment
        self.total_slices = 512 * len(os.listdir(self.img_dir))
        self.labels=labels
        self.factor=factor

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
    def split_v2(self):
        self.folder_mapping=[]
        self.folder_dict={}
        i=0
        l1,l2,l3,l4=0,0,0,0
        for folder in os.listdir(self.img_dir):
            t=os.path.join(self.img_dir,folder)
            original_imgs,_=nrrd.read(os.path.join(t,'CT-vol.nrrd'))
            labels,_=nrrd.read(os.path.join(t,'Segmentation-label.nrrd'))
            labels[np.where(labels == 3)] = 0
            labels[np.where(labels == 4)] = 3
            t1, t2, t3, t4 = np.where(labels == 0), np.where(labels == 1), np.where(labels == 2), np.where(labels == 3)
            l1+=labels[t1].size
            l2+=labels[t2].size
            l3+=labels[t3].size
            l4+=labels[t4].size
            self.folder_mapping.append(original_imgs.shape[-1])
            self.folder_dict[i]=original_imgs.shape[-1]//self.patch_depth if original_imgs.shape[-1]%self.patch_depth==0 else original_imgs.shape[-1]//self.patch_depth+1
            if original_imgs.shape[-1]%self.patch_depth!=0:
                ex=original_imgs.shape[-1]%self.patch_depth
                ex=self.patch_depth-ex
                temp=labels[:,:,-1]
                l1 += ex*temp[np.where(temp==0)].size
                l2 += ex*temp[np.where(temp==1)].size
                l3 += ex*temp[np.where(temp==2)].size
                l4 += ex*temp[np.where(temp==3)].size
            i+=1
        self.all_patch_number=sum(self.folder_dict.values())
        self.all_index=np.array(range(self.all_patch_number))
        val_len=int(self.split_rate*len(self.all_index))
        self.val_index=np.random.choice(self.all_index,size=val_len,replace=False)
        self.train_index=np.array([i for i in self.all_index if i not in self.val_index])
        self.steps_per_epoch=len(self.train_index)*self.factor*self.factor//self.train_bs
        self.valid_steps=len(self.val_index)*self.factor*self.factor//self.val_bs
        print('val len: {}, train len: {}  all patch number:{}'.format(len(self.val_index), len(self.train_index),
                                                                   len(self.val_index) + len(self.train_index)))
        l=l1+l2+l3+l4
        return l/l1,l/l2,l/l3,l/l4
    def find_folder_and_patch_id(self,pid):
        res=0
        for i in self.folder_dict:
            res+=self.folder_dict[i]
            if res>pid:
                return i,pid-(res-self.folder_dict[i])
        raise Exception('dict error')
    def generator_v2(self,valid=False):
        index=self.train_index if valid==False else self.val_index
        batch_size=self.train_bs if valid==False else self.val_bs
        x=[]
        y=[]
        count=0
        if self.augment==False:
            while True:
                np.random.shuffle(index)
                for i in index:
                    folder_id, patch_id = self.find_folder_and_patch_id(i)
                    # patch_id=self.folder_dict[folder_id]-1
                    folder_path=os.path.join(self.img_dir,os.listdir(self.img_dir)[folder_id])
                    original_imgs,_=nrrd.read(os.path.join(folder_path,'CT-vol.nrrd'))
                    labels,_=nrrd.read(os.path.join(folder_path,'Segmentation-label.nrrd'))
                    original_imgs=np.array([original_imgs[:,:,i] for i in range(original_imgs.shape[-1])])
                    labels=np.array([labels[:,:,i] for i in range(labels.shape[-1])])
                    #标签的顺序应该是background=0, nerve=1, bone=2, vessel=3（如果有vessel）, disc=4, vessel 改成bg0, disc改成3
                    #切割
                    if (patch_id+1)*self.patch_depth<=original_imgs.shape[0]:
                        patch_x=original_imgs[patch_id*self.patch_depth:(patch_id+1)*self.patch_depth,:,:]
                        patch_y=labels[patch_id*self.patch_depth:(patch_id+1)*self.patch_depth,:,:]
                    else:
                        """
                        切割到了尽头处，有不够的部分,用最后一个平面重复补充
                        """
                        ex=self.folder_mapping[folder_id]%self.patch_depth
                        ex=self.patch_depth-ex
                        patch_x=original_imgs[patch_id*self.patch_depth:,:,:]
                        patch_y=labels[patch_id*self.patch_depth:,:,:]
                        patch_x = np.append(patch_x, ex*[original_imgs[-1, :, :]], axis=0)
                        patch_y = np.append(patch_y, ex*[labels[-1, :, :]], axis=0)
                    patch_x=np.expand_dims(patch_x,axis=-1)
                    patch_y[np.where(patch_y==3)]=0
                    patch_y[np.where(patch_y==4)]=3
                    print(np.unique(patch_y))
                    shape=patch_y.shape
                    patch_y = np.eye(self.labels)[patch_y.reshape(-1)].reshape((shape[0],shape[1],shape[2],self.labels))
                    #4,512,512,1   4 512 512 5
                    # x.append(patch_x)
                    # y.append(patch_y)
                    delta=shape[1]//self.factor
                    for h in range(self.factor):
                        for w in range(self.factor):
                            x.append(patch_x[:,h*delta:(h+1)*delta,w*delta:(w+1)*delta,:])
                            y.append(patch_y[:, h * delta:(h + 1) * delta, w*delta:(w + 1) * delta, :])
                            count+=1
                            if count >= batch_size:
                                x, y = np.array(x), np.array(y)
                                # print(x.shape, y.shape)
                                yield x, y
                                count = 0
                                x = []
                                y = []
        else:
            """
            数据增强函数
            """
            pass
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
    d=DataGenerator('C:\\Users\chris.li2\\3D_medical')
    d.split_v2()
    d.generator_v2()
    d.generator_v2().__next__()