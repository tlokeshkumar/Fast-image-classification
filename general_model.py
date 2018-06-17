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
from common_network import preprocess_input, transfer_model, create_non_trainable_model
import math
# import libaries
import pickle
import json
import datetime
import numpy as np
import cv2
import os, sys
from tqdm import tqdm
from glob import glob
import shutil
from tqdm import tqdm
import csv
import tensorflow as tf
import argparse
import random
import h5py
#######################DOCUMENTATION#####################
# To change the Bottleneck tensor, change the bottleneck tensorname variable anf if you want, transfer learning model
# Check for the dimension in load_cached_bottlenecks function and resolve any shape conflicts whatsoever
# Sample Execution
# If you want to create bottlenecks (i.e) running for the first time
# python resnet_transfer.py --train train_dir --val val_dir --logs resnet50_logs --create_bottleneck --bottleneck_dir bottleneck_flowers
# If bottlenecks have been created and you just want to train the deep neural network, just remove the create_bottleneck flag
# python resnet_transfer.py --train train_dir --val val_dir --logs resnet50_logs  --bottleneck_dir bottleneck_flowers
#
# Beware of Memory errros, if you are saving intermediate bottlenecks, you can end up in saving a lot of floating
# point numbers that the bottlenecks size can become mush more than the real dataset size
#
# Sample Execution
# python resnet_transfer.py --train train_dir --val val_dir --logs resnet50_logs --bottleneck_dir bottleneck_flowers
# To omit bottlenecks and train normally,
# python resnet_transfer.py --train train_dir --val val_dir --logs resnet50_logs --bottleneck_dir bottleneck_flowers --omit_bottlenecks
###########################################################

parser = argparse.ArgumentParser()
parser.add_argument("--train", help = "Where the dataset files are present")
parser.add_argument("--val", help = "Enter the path to the validation set")
parser.add_argument("--logs", help = "Where the log files should be saved")
parser.add_argument("--create_bottleneck", action ='store_true', help = "Enter whether bottlenecks were created for the directory specified or not")
parser.add_argument("--bottleneck_dir", help = "Enter path where you want to store the bottlenecks")
parser.add_argument("--log_dir", help  = "Place to write the log files")
parser.add_argument("--omit_bottlenecks", action = "store_true",help = "Start normal training without without calculating bottlenecks")
parser.add_argument("--load_weights", help = "Enter path where the trained weights are present")
parser.add_argument("--bottleneck_tensorname", help = "Enter the layer of the pre-trained network which you want to make as the bottleneck")
parser.add_argument("--base_model", choices = ['vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'inception_resnetv2', 'xception', 
'densenet121', 'densenet169', 'densenet201', 'nasnetmobile', 'nasnetlarge'], default = 'vgg16', help = 'Enter the network you want as your base feature extractor')
parser.add_argument("--batch_size_train", default = 128,type = int, help = 'Enter the batch size that must be used to train')
parser.add_argument('--epochs', default = 100,type = int, help = 'Enter the number of epochs to train')
parser.add_argument('--bottlenecks_batch_size', default = 32,type = int, help = 'Enter the batch size to create the bottlenecks. Only relavant if you are creating bottlenecks')
parser.add_argument('--saving_ckpts', default = 1,type = int, help = 'When do you want to store model during training time? (in number of epochs')
parser.add_argument('--weight_file', default = 'top.h5', help = 'The name of the weight file what will be stored. (*.h5)')
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

## load data

LABEL_LENGTH = len(glob(args.train + '/*'))
# BOTTLENECK_TENSOR_NAME = 'activation_31'
BATCH_SIZE = args.batch_size_train
BOTTLENECKS_BATCHSIZE = args.bottlenecks_batch_size
EPOCHS = args.epochs

# function to read image
def chunks(l, n):
	c = []
	for i in range(0, len(l), n):
		c.append(l[i:i + n])
	return c
def create_single_bottleneck(phase,image_addr,non_trainable_model):
	bottleneck_train_dir = args.bottleneck_dir + "/" +phase	
	img = cv2.imread(image_addr)
	img = cv2.resize(img, (256,256))
	img = np.expand_dims(img, axis = 0)
	# dummy_model = Model(inputs = non_trainable_model.input, outputs = non_trainable_model.get_layer(BOTTLENECK_TENSOR_NAME).output)
	dummy_model = Model(inputs = non_trainable_model.input, outputs = non_trainable_model.output)
	# bottleneck_features_train  = non_trainable_model.predict(img)
	bottleneck_features_train  = dummy_model.predict(img)
	bottleneck_features_train = np.squeeze(bottleneck_features_train)
	image_name = os.path.split(image_addr)[1]
	class_name = os.path.split(os.path.split(image_addr)[0])[1] # Getting the class name of the example
	np.save(bottleneck_train_dir + "/" + class_name + "/" +image_name+".npy", bottleneck_features_train)
	return bottleneck_train_dir + "/" + class_name + "/" +image_name+".npy"
	# npy_dirs.append(bottleneck_train_dir + "/" + class_name + "/" +image_name + ".npy")
	# npy_labels.append(class_name)
	# addr_label_map = dict(zip(npy_dirs, npy_labels))
def multiprocess_bottleneck_creation(phase, label_map, dataset, non_trainable_model):
	with concurrent.futures.ProcessPoolExecutor() as executor:
		img_addrs = glob(dataset+"/**/*")
		for img_addr, npy_file in tqdm(zip(img_addrs, executor.map(create_single_bottleneck, (phase, img_addrs, non_trainable_model)))):
			img_addr = True
def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256,256))
    return img
def create_bottlenecks_h5py(phase, label_map, dataset, non_trainable_model):
	'''
	Parameters
	----------
	phase: whether training ('train') or evaluation('val') (validation phase)
	dataset: path where the dataset is present
	non_trainable_model: the part of the model where the training doesnot happen
	label_map: The map which contains the label and the corresponding index

	Returns
	-------
	Saves all the bottlenecks of the corresponding images in corresponding directories
	addr_label_map: dictionary called labels where for each ID of the dataset, the associated
			label is given by labels[ID] (ID refers to path of each bottleneck (.npy file))
	npy_dirs: A list of all addrs of npy files that are saved
	'''
	bottleneck_train_dir = args.bottleneck_dir + "/" +phase
	# Creating the directory heirarchy incase if its not existant before
	if not os.path.exists(args.bottleneck_dir):
		os.mkdir(args.bottleneck_dir )
	if not os.path.exists(bottleneck_train_dir):
		os.mkdir(bottleneck_train_dir)
	# for i in list(label_map.keys()):
	# 	if not os.path.exists(bottleneck_train_dir + "/" +i):
	# 		os.mkdir(bottleneck_train_dir + "/" +i)
	f = h5py.File(bottleneck_train_dir+'/'+phase+'.h5', 'w')
	print ("[INFO] Creating " + phase + " Bottlenecks")
	# bottleneck_features_train = non_trainable_model.predict_generator(h1, predict_size_train, verbose = 1)
	image_addrs = glob(dataset + "/**/*.jpg")
	h5py_dirs = []
	h5py_labels = []
	batch_addrs = chunks(image_addrs, BOTTLENECKS_BATCHSIZE)
	for chunk in tqdm(batch_addrs):
		img_batch = []
		for image_addr in (chunk):
			img = cv2.imread(image_addr)
			img = cv2.resize(img, (256,256))
			img_batch.append(img)
			# img = np.expand_dims(img, axis = 0)
			# dummy_model = Model(inputs = non_trainable_model.input, outputs = non_trainable_model.get_layer(BOTTLENECK_TENSOR_NAME).output)
		dummy_model = Model(inputs = non_trainable_model.input, outputs = non_trainable_model.output)
		# bottleneck_features_train  = non_trainable_model.predict(img)
		bottleneck_features_train  = dummy_model.predict(np.array(img_batch))
		# bottleneck_features_train = np.squeeze(bottleneck_features_train)
		for idx,image_addr in enumerate(chunk):
			# image_name = os.path.split(image_addr)[1]
			class_name = os.path.split(os.path.split(image_addr)[0])[1] # Getting the class name of the example
			# np.save(bottleneck_train_dir + "/" + class_name + "/" +image_name+".npy", bottleneck_features_train)
			f.create_dataset(image_addr, data = bottleneck_features_train[idx])
			h5py_dirs.append(image_addr)
			h5py_labels.append(class_name)
			addr_label_map = dict(zip(h5py_dirs, h5py_labels))
	return (addr_label_map, h5py_dirs, bottleneck_train_dir+'/'+phase+'.h5')
def create_bottlenecks(phase, label_map, dataset, non_trainable_model):
	bottleneck_train_dir = args.bottleneck_dir + "/" +phase
	# Creating the directory heirarchy incase if its not existant before
	if not os.path.exists(args.bottleneck_dir):
		os.mkdir(args.bottleneck_dir )
	if not os.path.exists(bottleneck_train_dir):
		os.mkdir(bottleneck_train_dir)
	for i in list(label_map.keys()):
		if not os.path.exists(bottleneck_train_dir + "/" +i):
			os.mkdir(bottleneck_train_dir + "/" +i)
	print ("[INFO] Creating " + phase + " Bottlenecks")
	# bottleneck_features_train = non_trainable_model.predict_generator(h1, predict_size_train, verbose = 1)
	image_addrs = glob(dataset + "/**/*.jpg")
	npy_dirs = []
	npy_labels = []
	for image_addr in tqdm(image_addrs):
		img = cv2.imread(image_addr)
		img = cv2.resize(img, (256,256))
		img = np.expand_dims(img, axis = 0)
		# dummy_model = Model(inputs = non_trainable_model.input, outputs = non_trainable_model.get_layer(BOTTLENECK_TENSOR_NAME).output)
		dummy_model = Model(inputs = non_trainable_model.input, outputs = non_trainable_model.output)
		# bottleneck_features_train  = non_trainable_model.predict(img)
		bottleneck_features_train  = dummy_model.predict(img)
		bottleneck_features_train = np.squeeze(bottleneck_features_train)
		image_name = os.path.split(image_addr)[1]
		class_name = os.path.split(os.path.split(image_addr)[0])[1] # Getting the class name of the example
		np.save(bottleneck_train_dir + "/" + class_name + "/" +image_name+".npy", bottleneck_features_train)
		npy_dirs.append(bottleneck_train_dir + "/" + class_name + "/" +image_name + ".npy")
		npy_labels.append(class_name)
	addr_label_map = dict(zip(npy_dirs, npy_labels))
	return (addr_label_map, npy_dirs)
def create_npy_class_map(phase, args):
	bottleneck_train_dir = args.bottleneck_dir + "/" +phase
	npy_dirs = glob(bottleneck_train_dir+"/**/*")
	npy_labels = []
	for i in range(npy_dirs):
		class_name = os.path.split(os.path.split(i)[0])[1]
		npy_labels.append(class_name)
	addr_label_map = dict(zip(npy_dirs, npy_labels))
	return addr_label_map, npy_dirs
def load_random_cached_bottlenecks(batch_size, label_map, addr_label_map, dirs, comp_type = 'h5py', hdf5_file = None):
	'''
	Parameters
	----------
	batch_size: Number of bottlenecks to be loaded along with the labels
	label_map: The dictionary that maps the class_names and the index
	addr_label_map: The dictionary that maps addrs of bottlenecks and the labels
	npy_dirs: A list of all the npy_dirs in the dataset
	Returns
	-------
	batch: (bottlenecks_train, bottlenecks_labels) a batch of them which is equal to batch_size
	'''
	if comp_type == 'npy':
		length_of_dataset = len(addr_label_map.keys())
		batch_index = np.random.randint(length_of_dataset, size = batch_size)
		chosen_npy = [dirs[i] for i in batch_index]
		labels_for_chosen_npy = [label_map[addr_label_map[i]] for i in chosen_npy]
		npy_data = np.array([(np.load(i))for i in chosen_npy])
		npy_onehot = to_categorical(labels_for_chosen_npy, num_classes = LABEL_LENGTH)
		return ((npy_data, npy_onehot))
	elif comp_type == 'h5py':
		length_of_dataset = len(addr_label_map.keys())
		batch_index = np.random.randint(length_of_dataset, size = batch_size)
		chosen_h5py = [dirs[i] for i in batch_index]
		labels_for_chosen_h5py = [label_map[addr_label_map[i]] for i in chosen_h5py]
		h5py_data = np.array([hdf5_file[i] for i in chosen_h5py])
		h5py_onehot = to_categorical(labels_for_chosen_h5py, num_classes = LABEL_LENGTH)
		return ((h5py_data, h5py_onehot))
def train_with_bottlenecks(args, label_map, trainable_model, non_trainable_model, iterations_per_epoch_t, iterations_per_epoch_v):
	if args.create_bottleneck:
		training_addr_label_map, train_npy_dir, h5py_file_train = create_bottlenecks_h5py("train", label_map, args.train, non_trainable_model)
		# multiprocess_bottleneck_creation("train", label_map, args.train, non_trainable_model)
		# training_addr_label_map, train_npy_dir = create_npy_class_map("train", args)
		# Writing the dictionaries to a txt file so that we neednt loop again in future
		with open("essential_files/train_addr_label_map.txt" , "wb") as file:
			pickle.dump(training_addr_label_map, file)
		with open("essential_files/train_npy_dir.txt", "wb") as file:
			pickle.dump(train_npy_dir, file)
	if not args.create_bottleneck:
		with open("essential_files/train_addr_label_map.txt" , "rb") as file:
			print ("[INFO] (Training)Loading Address to Label Map from Disk")
			training_addr_label_map = pickle.load(file)
		with open("essential_files/train_npy_dir.txt", "rb") as file:
			print ("[INFO] (Training)Loading Address from Disk")
			train_npy_dir = pickle.load(file)

	# Saving the bottleneck features for the bottom nontrainable model (validation dataset)

	# Creating bottlenecks if its not created
	if args.create_bottleneck:
		validation_addr_label_map, val_npy_dir, h5py_file_val = create_bottlenecks_h5py("val", label_map, args.val, non_trainable_model)
		# multiprocess_bottleneck_creation("val", label_map, args.val, non_trainable_model)
		# validation_addr_label_map, val_npy_dir = create_npy_class_map("val", args)
		with open("essential_files/validation_addr_label_map.txt", "wb") as file:
			pickle.dump(validation_addr_label_map, file)
		with open("essential_files/val_npy_dir.txt", "wb") as file:
			pickle.dump(val_npy_dir, file)
	if not args.create_bottleneck:
		with open("essential_files/validation_addr_label_map.txt", "rb") as file:
			print ("[INFO] (Validation)Loading Address to Label Map from Disk")
			validation_addr_label_map = pickle.load(file)
		with open("essential_files/val_npy_dir.txt", "rb") as file:
			print ("[INFO] (Validation)Loading Address to Label Map from Disk")
			val_npy_dir = pickle.load(file)

	print ("[INFO] Loading the bottlenecks")

	print ("[INFO] Starting to Train")
	history_information=[]
	h5py_file_train =args.bottleneck_dir+'/train/train'+'.h5'
	h5py_file_val =args.bottleneck_dir+'/val/val'+'.h5'
	h5py_file_train = h5py.File(h5py_file_train, 'r')
	h5py_file_val = h5py.File(h5py_file_val, 'r')
	print (trainable_model.summary())
	for epoch in range(EPOCHS):
		for i in range(iterations_per_epoch_t*EPOCHS):
			X,Y = load_random_cached_bottlenecks(BATCH_SIZE, label_map, training_addr_label_map, train_npy_dir, 'h5py', h5py_file_train)
			
			loss = trainable_model.train_on_batch(X, Y)
			history_information.append(loss)
			if i%10 == 0:
				print (str(datetime.datetime.now())+"\tPercent to complete: " + str((iterations_per_epoch_t*EPOCHS - i)*100//(iterations_per_epoch_t*EPOCHS))+"\t\tEpoch: " + str(epoch) + "\tIteration: " + str(i) + '\tLoss: ' + str(loss[0]) + "\tTraining_Accuracy: " + str(loss[1]))
		if epoch% args.saving_ckpts == 0:
			trainable_model.save(args.weight_file)
		for i in range(iterations_per_epoch_v):
			X,Y = load_random_cached_bottlenecks(BATCH_SIZE, label_map, validation_addr_label_map, val_npy_dir, 'h5py', h5py_file_val)
			loss = trainable_model.test_on_batch(X, Y)
			print ("\tIteration: " + str(i) + '\tLoss: ' + str(loss[0]) + "\tValidation_Accuracy: " + str(loss[1]))
	np.save("essential_files/history_training.npy", np.array(history_information))
	print ("[INFO] Completed Training!")
def train_without_bottlenecks(train_generator, val_generator, model, train_step_epoch = 2000, val_step_epoch = 500, callback_list = []):
	'''
	Parameters
	----------
	train_generator: This is the train data generator (flow from directory)
	val_generator: validation data generator which generates validation data in batches
	model: This is the trainable model that bypasses generation of bottlenecks
	train_step_epoch: No of steps in training phase for one epoch to complete
	val_step_epoch: No of steps in validation phase for one epoch (validation data) to complete
	callbacks: Optional list of callbacks needed.
	
	Returns
	-------
	history: Record of training values and training accuracies
	'''
	history = model.fit_generator(
		train_generator,
		steps_per_epoch = train_step_epoch,
		validation_steps = val_step_epoch,
		epochs = EPOCHS,
		validation_data = val_generator,
		callbacks = callback_list
	)
	return history
def trainable_model(non_trainable_model):
	'''
	Parameters
	----------
	non_trainable_model: This is the non_trianable base network whose output features are used to trian this
	network.

	Returns
	-------
	trainable_model: The fine_tuning model object that we will train.
	'''
	input_tensor = Input(shape = non_trainable_model.output_shape[1:])
	# input_tensor = Input(shape = non_trainable_model.get_layer(BOTTLENECK_TENSOR_NAME).output_shape[1:])
	# weights = non_trainable_model.get_weights()
	trainable_model = transfer_model(input_tensor, LABEL_LENGTH)
	# trainable_model.summary()
	return (trainable_model)

# for i in (base_model.layers):
	# print (i.name)
# exit(0)
non_trainable_model = create_non_trainable_model(base_model, BOTTLENECK_TENSOR_NAME)
trainable_model = trainable_model(non_trainable_model)


# Train data generator
train_datagen = ImageDataGenerator(
		preprocessing_function = preprocess_input)

# Callback list
checkpoint = ModelCheckpoint('ResNet50-transferlearning.h5')
tb_callback = keras.callbacks.TensorBoard(
	log_dir=args.logs,
	histogram_freq=2,
	write_graph=True
)
# early_stopping = EarlyStopping(monitor = 'val_loss')
callback_list = [checkpoint, tb_callback]#, early_stopping]


h1 = train_datagen.flow_from_directory(args.train, batch_size=BATCH_SIZE)

# Predicting the size of triain and number of steps
nb_train_samples = len(h1.filenames)
num_classes = len(h1.class_indices)
predict_size_train = int(math.ceil(nb_train_samples / BATCH_SIZE))

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
validation_generator = test_datagen.flow_from_directory(
		args.val,
		batch_size=16,
		)
nb_validation_samples = len(validation_generator.filenames)  
predict_size_validation = int(math.ceil(nb_validation_samples / BATCH_SIZE))  

trainable_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
			metrics=['accuracy'])
iterations_per_epoch_t = math.ceil((nb_train_samples*1.0/BATCH_SIZE))
iterations_per_epoch_v = math.ceil((nb_validation_samples*1.0/BATCH_SIZE))

# Creating a label map and storing them
label_map = (h1.class_indices)
inv_label_map = dict((v,k) for k,v in label_map.items())
if not os.path.exists("essential_files"):
	os.mkdir("essential_files")
with open("essential_files/label_map.json", "w") as file:
	json.dump(label_map, file)

# saving the bottleneck features for the bottom nontrainable model if not created earlier
if not args.omit_bottlenecks:
	non_trainable_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),metrics=['accuracy'])
	non_trainable_model.summary()
	if args.load_weights is not None:
		non_trainable_model.load_weights(args.load_weights)
	train_with_bottlenecks(args, label_map, trainable_model, non_trainable_model, iterations_per_epoch_t, iterations_per_epoch_v)
else:
	non_trainable_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),metrics=['accuracy'])
	if args.load_weights is not None:
		# non_trainable_model = load_model(args.load_weights)
		non_trainable_model.load_weights(args.load_weights)
	# non_trainable_model.summary()
	train_without_bottlenecks(h1, validation_generator, non_trainable_model, iterations_per_epoch_t, iterations_per_epoch_v, callback_list)

# if __name__ == '__main__':
# 	main(args, base_model)