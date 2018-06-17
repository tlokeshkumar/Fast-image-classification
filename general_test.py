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
from common_network import create_non_trainable_model, preprocess_input, transfer_model
# import libaries
import json
from natsort import natsorted
import numpy as np
import cv2
import os, sys
from glob import glob
import argparse
##############################DOCUMENTATION#####
# For testing the function in transfer_model should be same (In training program) so that we can load the trained weights properly.
# Note also BOTTLENECK_TENSORNAME must be same in both the cases , training and testing

parser = argparse.ArgumentParser()
parser.add_argument("--weight_file", default = 'top.h5',help = "Enter the path where the weight file is stored")
parser.add_argument("--label_file", help = "Enter the path where the label_map is stored")
parser.add_argument("--img_dir", help = "Enter the path of the dir where images are stored for prediction")
parser.add_argument("--base_model", choices = ['vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'inception_resnetv2', 'xception', 
'densenet121', 'densenet169', 'densenet201', 'nasnetmobile', 'nasnetlarge'], default = 'vgg16', help = 'Enter the network you want as your base feature extractor')
parser.add_argument("--bottleneck_tensorname", help = "Enter the layer of the pre-trained network which you want to make as the bottleneck")
args = parser.parse_args()

if args.base_model == 'vgg16':
	base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'vgg19':
	base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'resnet50':
	base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'inceptionv3':
	base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'inception_resnetv2':
	base_model = applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'xception':
	base_model = applications.Xception(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'densenet121':
	base_model = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'densenet169':
	base_model = applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'densenet201':
	base_model = applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'nasnetmobile':
	base_model = applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
elif args.base_model == 'nasnetlarge':
	base_model = applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
else:
	raise ValueError("Model you entered is not present in the model zoo thats offered")

if args.bottleneck_tensorname is  None:
	# Taking the last tensor (Just before the softmax layer)
	BOTTLENECK_TENSOR_NAME = base_model.layers[-1].name
else:
	BOTTLENECK_TENSOR_NAME = args.bottleneck_tensorname

with open(args.label_file, "r") as file:
    label_map = json.load(file)
inv_label_map = dict((v,k) for k,v in label_map.items())

LABEL_LENGTH = len(label_map)
def predict_complete_output(non_trainable_model, trainable_model, img, label_map, want_probability = False):
    '''
    Parameters
    ----------
    non_trainable_model: This is the model which is used for feature (informally bottleneck) extraction
    trainable_model: This is the model which was trained for fine tuning
    img: Image which you want to predict
    label_map: The dictionary that maps index and class_names
    want_probability: This if made True will return the probabilities predicted by the network
    Returns
    ------
    predictions: The probability outputs of the network for the given image
    '''
    feature_extracted = non_trainable_model.predict(img)
    print (feature_extracted.shape)
    predictions = trainable_model.predict(feature_extracted)
    class_name = label_map[np.argmax(predictions[0])]
    if want_probability:
        return (class_name, predictions[0])
    else:
        return (class_name, predictions[0][np.argmax(predictions[0])])
# base_model = applications.ResNet50(weights = 'imagenet', include_top = False, input_shape = (256,256,3))
input_tensor = Input(shape = base_model.get_layer(BOTTLENECK_TENSOR_NAME).output_shape[1:])
trainable_model = transfer_model(input_tensor, LABEL_LENGTH)
trainable_model = load_model(args.weight_file)
non_trainable_model = create_non_trainable_model(base_model, BOTTLENECK_TENSOR_NAME)
print (trainable_model.input_shape)
for img_addr in natsorted(glob(args.img_dir + "/*")):
	img = cv2.imread(img_addr)
	img = cv2.resize(img, (256,256))
	img = np.expand_dims(img, axis = 0)
	(class_name, confidence) = predict_complete_output(non_trainable_model, trainable_model, img, inv_label_map)
	print ((class_name, confidence))


