from keras import backend as K
from functools import partial
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

class loss_metrics(object):
    def __init__(self,smooth=1.0):
        self.smooth=smooth
    def dice_coef_loss(self,y_true,y_pred):
        y_true=K.flatten(y_true)
        y_pred=K.flatten(y_pred)
        intersection=K.sum(y_true*y_pred)
        return -(2*intersection+self.smooth)/(K.sum(y_true)+K.sum(y_pred)+self.smooth)
    def weighted_dice_coef_loss(self,y_true,y_pred):
        return -K.mean(2. * (K.sum(y_true * y_pred) + self.smooth / 2) / (K.sum(y_true) + K.sum(y_pred) + self.smooth))
    def label_wise_dice_metrics(self,y_true,y_pred,label_index):
        return -self.dice_coef_loss(y_true[:,:,:,:,label_index],y_pred[:,:,:,:,label_index])
    def get_label_wise_dice_mertrics(self,label_index):
        f=partial(self.label_wise_dice_metrics,label_index=label_index)
        f.__setattr__('__name__', 'label_{0}_dice_mertrics'.format(label_index))
        return f
    def softmax_entropy_acc(self,y_true,y_pred):
        return K.cast(K.equal(K.argmax(y_true, axis=-1),
                              K.argmax(y_pred, axis=-1)),
                      K.floatx())
    def label_wise_softmax_entropy_acc(self,y_true,y_pred,label_index):
        return self.softmax_entropy_acc(y_true[:,:,:,:,label_index],y_pred[:,:,:,:,label_index])
    def get_label_wise_crossentropy_acc(self,label_index):
        f=partial(self.label_wise_softmax_entropy_acc,label_index=label_index)
        f.__setattr__('__name__', 'label_{0}_crossentropy_mertrics'.format(label_index))
        return f








