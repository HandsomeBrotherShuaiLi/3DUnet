from keras.engine import Input,Model
from keras.layers import Conv3D,MaxPooling3D,UpSampling3D,Activation
from keras.layers import BatchNormalization,PReLU,Deconvolution3D
from keras.layers.merge import concatenate
"""
3D U-Net,注意3D卷积的输入tensor维度!!!! 后面再做GAN
(batch, conv_dim1, conv_dim2, conv_dim3, channels)
"""
class UNet(object):
    def __init__(self,input_shape,label_numbel,depth=4,n_base_filters=32,
                 batch_normalization=True,deconvolution=False,pool_size=(2,2,2)):
        self.input_shape=input_shape
        self.label_number=label_numbel
        self.depth=depth
        self.n_base_filters=n_base_filters
        self.batch_normalization=batch_normalization
        self.deconvolution=deconvolution
        self.pool_size=pool_size

    def convolution_block(self,name,input_layer,n_filters,batch_normalization=True,
                          kernel=(3,3,3),strides=(1,1,1)):
        layer=Conv3D(n_filters,kernel,strides=strides,padding='same',kernel_initializer='he_normal',name=name+'_conv')(input_layer)
        if batch_normalization:
            #channels last format axis=-1
            layer=BatchNormalization(name=name+'_bn')(layer)
        layer=Activation('relu')(layer)
        return layer

    def up_convolution_block(self,name,input_layer,n_filters,pool_size,
                             kernel_size=(2,2,2),strides=(2,2,2),
                             deconvolution=False):
        if deconvolution:
            return Deconvolution3D(filters=n_filters,kernel_size=kernel_size,strides=strides,
                                   kernel_initializer='he_normal',name=name+'_deconv')(input_layer)
        return UpSampling3D(size=pool_size,name=name+'_Upsampling')(input_layer)

    def model(self):
        input_layer=Input(shape=self.input_shape)
        current_layer=input_layer
        level=[]
        for layer_depth in range(self.depth):
            layer1=self.convolution_block(
                input_layer=current_layer,n_filters=self.n_base_filters*(2**layer_depth),
                batch_normalization=self.batch_normalization,name='layer1_stage1_{}'.format(str(layer_depth))
            )
            layer2=self.convolution_block(
                input_layer=layer1,n_filters=self.n_base_filters*(2**(layer_depth+1)),
                batch_normalization=self.batch_normalization,name='layer2_stage1_{}'.format(str(layer_depth))
            )
            if layer_depth<self.depth-1:
                current_layer=MaxPooling3D(pool_size=self.pool_size)(layer2)
                level.append([layer1,layer2])
            else:
                current_layer=layer2
                level.append([layer1,layer2])
        for layer_depth in range(self.depth-2,-1,-1):
            #channel last format!!!!!!!
            #bs,n,w,h,c
            #bs,63,w,h,1
            up_convolution=self.up_convolution_block(
                pool_size=self.pool_size,deconvolution=self.deconvolution,n_filters=current_layer.get_shape()[-1],
                input_layer=current_layer,name='up_conv_stage2_{}'.format(str(layer_depth))
            )
            concat_layer=concatenate([up_convolution,level[layer_depth][1]])
            current_layer=self.convolution_block(
                n_filters=int(level[layer_depth][1].get_shape()[-1]),input_layer=concat_layer,
                batch_normalization=self.batch_normalization,name='conv_stage2.1_{}'.format(str(layer_depth))
            )
            current_layer=self.convolution_block(
                n_filters=int(level[layer_depth][1].get_shape()[-1]),input_layer=current_layer,
                batch_normalization=self.batch_normalization,name='conv_stage2.2_{}'.format(str(layer_depth))
            )
        #多类别分类用softmax,多标签二分类(Yes or No) 用sigmoid
        prediction=Conv3D(self.label_number,(1,1,1),activation='softmax',kernel_initializer='he_normal',name='prediction')(current_layer)
        model=Model(input_layer,prediction)
        model.summary()
        return model

if __name__=="__main__":
    m=UNet(
        input_shape=(512,512,None,1),
        label_numbel=5
    )
    m.model()




