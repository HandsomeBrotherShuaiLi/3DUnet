from keras.engine import Input,Model
from keras.layers import Conv3D,MaxPooling3D,UpSampling3D,Activation
from keras.layers import BatchNormalization,Deconvolution3D
from keras.layers.merge import concatenate
from model.data_generator import DataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import TensorBoard,ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
import os
from keras import backend as K
from model.loss import loss_metrics
"""
3D U-Net,注意3D卷积的输入tensor维度!!!! 后面再做GAN
(batch, conv_dim1, conv_dim2, conv_dim3, channels)
tf.nn.
(batch, Depth, H, W, channels)
(batch, depth, height, width, channels)
reference: https://www.tensorflow.org/api_docs/python/tf/layers/Conv3D
"""
def top_k_categorical_accuracy(y_true, y_pred, k=4):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)

class weighted_categorical_crossentropy(object):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        loss = weighted_categorical_crossentropy(weights).loss
        model.compile(loss=loss,optimizer='adam')
    """

    def __init__(self, weights):
        self.weights = K.variable(weights)

    def loss(self, y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred,axis=-1, keepdims=True)
        # clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # calc
        loss = y_true * K.log(y_pred) * self.weights
        loss = -K.sum(loss, -1)
        return loss

class UNet(object):
    def __init__(self,input_shape,label_numbel=5,depth=4,n_base_filters=32,
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
            #bs,w,h,n,c
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
    def train(self,img_dir,train_bs=1,val_bs=1,opt='adam',augment=False,split_rate=0.1,lr=1e-3,model_folder='D:\py_projects\\3DUnet\models',factor=4):
        """
        还是要reshape的，主要是concat 那里不方便
        :param img_dir:
        :param train_bs:
        :param val_bs:
        :param opt:
        :param augment:
        :param split_rate:
        :param lr:
        :param model_folder:
        :return:
        """
        d=DataGenerator(img_dir=img_dir,train_bs=train_bs,val_bs=val_bs,patch_depth=self.input_shape[0],
                                 shape=None if self.input_shape[1:]==(None,None,1) else self.input_shape[1:],
                                 labels=self.label_number,split_rate=split_rate,augment=augment,factor=factor)
        f1,f2,f3,f4=d.split_v2()
        train_steps_per_epoch,vaild_steps=d.steps_per_epoch,d.valid_steps
        print(train_steps_per_epoch,vaild_steps)
        model=self.model()
        metrics_list=[loss_metrics.get_label_wise_crossentropy_acc(index) for index in range(self.label_number)]
        metrics_list.append('acc')
        dice_loss=loss_metrics.dice_coef_loss
        model.compile(optimizer=Adam(1e-3) if opt=='adam' else SGD(lr=lr,momentum=0.9,nesterov=True),
                      # loss='categorical_crossentropy',
                      loss=weighted_categorical_crossentropy([f1,f2,f3,f4]).loss,
                      metrics=metrics_list)
        his=model.fit_generator(
            generator=d.generator_v2(valid=False),
            steps_per_epoch=train_steps_per_epoch,
            validation_data=d.generator_v2(valid=True),
            validation_steps=vaild_steps,
            verbose=1,
            initial_epoch=0,
            epochs=100,
            callbacks=[
                TensorBoard('log',update_freq='batch'),
                EarlyStopping(monitor='val_loss',min_delta=0.00001,patience=20,verbose=1),
                ReduceLROnPlateau(monitor='val_loss',min_delta=1e-4,patience=5,verbose=1,factor=0.5),
                ModelCheckpoint(filepath=os.path.join(model_folder,'loss_lead_3D-UNet--{epoch:02d}--{val_loss:.5f}--{val_acc:.5f}.h5'),
                                monitor='val_loss',verbose=1,save_weights_only=False,save_best_only=True,period=1)
            ]
        )
        print(his.history)
if __name__=="__main__":
    m=UNet(
        input_shape=(8, None,None,1),
        label_numbel=4
    )
    m.train(img_dir='C:\\Users\chris.li2\\3D_medical',
            model_folder='D:\py_projects\\3DUnet\models',
            factor=4,
            train_bs=2,
            val_bs=4
            )




