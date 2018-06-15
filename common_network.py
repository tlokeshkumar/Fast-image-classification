import keras
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, concatenate, average
from keras.layers import Conv2D, MaxPooling2D, Input, Activation, add,GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.regularizers import l1, l2
import keras.backend as K
import numpy as np

def create_non_trainable_model(base_model, BOTTLENECK_TENSOR_NAME):
    '''
    Parameters
    ----------
    base_model: This is the pre-trained base model with which the non-trainable model is built

    Note: The term non-trainable can be confusing. The non-trainable-parametes are present only in this
    model. The other model (trianable model doesnt have any non-trainable parameters). But if you chose to 
    omit the bottlenecks due to any reason, you will be training this network only. (If you choose
    --omit_bottleneck flag). So please adjust the place in this function where I have intentionally made 
    certain layers non-trainable.

    Returns
    -------
    non_trainable_model: This is the model object which is the modified version of the base_model that has
    been invoked in the beginning. This can have trainable or non trainable parameters. If bottlenecks are
    created, then this network is completely non trainable, (i.e) this network's output is the bottleneck
    and the network created in the trainable is used for training with bottlenecks as input. If bottlenecks
    arent created, then this network is trained. So please use accordingly.
    '''
    # This post-processing of the deep neural network is to avoid memory errors
    x = GlobalAveragePooling2D()(base_model.get_layer(BOTTLENECK_TENSOR_NAME).output)
    print ("non trainable")
    print (x.shape)
    # If you want to further process the output of the non-trainable-base network please do here
    # Making all the layers non-trainable
    non_trainable_model = Model(inputs = base_model.input, outputs = [x])
    for layer in non_trainable_model.layers:
        layer.trainable = False
    # non_trainable_model = Model(inputs=base_model.input, outputs=[x])
    return (non_trainable_model)
def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[::-1, :, :]
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[ :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x
def transfer_model(bottleneck, LABEL_LENGTH, weights = None):
    '''
    Training starts from 'add_10' refer down in the base model split. So training(non trainable) till stage
    4, block 'c'. So creating the remaining resnet part thats from stage 4 block 'd' to end.
    '''
    # x = identity_block(bottleneck, 3, [256, 256, 1024], stage=4, block='d')
	# x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
	# x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

	# x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
	# x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
	# x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

	# x = AveragePooling2D((7, 7), name='avg_pool')(x)
	######
    # if len (bottleneck.shape) == 4:
    #     x = GlobalAveragePooling2D()(bottleneck)
		# x = Dense(257, kernel_regularizer = l2(0.1), activity_regularizer = l1(0.1))(x)
	# else:
		# x = Dense(257, kernel_regularizer = l2(0.1), activity_regularizer = l1(0.1))(bottleneck)
	########
    # Adding our own layer to get probability outputs
    # x = Dense(257, kernel_regularizer = l2(0.1), activity_regularizer = l1(0.1))(x)
    # x = Dropout(0.4)(x)
    # x = Activation('relu')(x)
    x = Dense(LABEL_LENGTH, kernel_regularizer = l2(0.05), activity_regularizer = l1(0.05))(bottleneck)
    # x = Dropout(0.4)(x)
    x = Activation('softmax')(x)
    model = Model(bottleneck, x, name = "Transfer_learning_model")
    # model.summary()
    # model.layers[-2].set_`weights([weights[-2], weights[-1]])
    # model.layers[-4].set_weights([weights[-4], weights[-3]])
    return (model)
