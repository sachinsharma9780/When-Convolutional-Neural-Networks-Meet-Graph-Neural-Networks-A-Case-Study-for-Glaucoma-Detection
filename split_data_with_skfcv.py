# import libraies
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import glob
import os
import cv2
from PIL import Image
from torchvision import transforms
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Split origa dataset into 10 folds using StratifiedKFold")
    parser.add_argument('-in', '--inpath', type=str, required=True, help='path to origa images already separated by two folders')
    parser.add_argument('-o', '--complete', type=str, required=True, help='path to all origa images in one folder')
    parser.add_argument('-fp', '--foldpath', type=str, required=True, help='path to store all 10 folds')
 

    return parser

#rcv inps from cmd
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

# read data
dataset = []
labels = []

# '/home/sachin/Desktop/Guided_research/origa/'
inp_path = args.inpath

inp_class0 = inp_path + 'class_0'
inp_class1 = inp_path + 'class_1'

# read img data in np fromat
def read_imgs(path, label):
	for file in os.listdir(path):
		dataset.append(file)
		labels.append(label)
		

# call
read_imgs(inp_class0, 0)
read_imgs(inp_class1, 1)


# check size of dataset and converting into numpy array
print('dataset length:', len(dataset))
dataset = np.array(dataset)
labels = np.array(labels)

# split data using stratifed kfcv and save all folds on disk
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)

# create directories for folds
def create_dirs(path):
	os.chdir(path)
	os.makedirs('train')
	os.makedirs('test')
	p1 = path + '/' + 'train' + '/'
	p2 = path + '/' + 'test' + '/'
	os.makedirs(os.path.join(p1, 'class_0'))
	os.makedirs(os.path.join(p1, 'class_1'))
	os.makedirs(os.path.join(p2, 'class_0'))
	os.makedirs(os.path.join(p2, 'class_1'))
	return p1, p2

# path to whole origa dataset 
#'/home/sachin/Desktop/Guided_research/origa_complete'
inp_path = args.complete

# copy imgs into respective folds
def copy_imgs(X_train, X_test, y_train, y_test, p1, p2):
	os.chdir(inp_path)
	x1 = p1 + 'class_0' + '/'
	x2 = p1 + 'class_1' + '/'
	x3 = p2 + 'class_0' + '/'
	x4 = p2 + 'class_1' + '/'
	for x, y in zip(X_train, y_train):
		if y == 0:
			shutil.copy(x, x1)
		else:
			shutil.copy(x, x2)

	for x, y in zip(X_test, y_test):
		if y == 0:
			shutil.copy(x, x3)	
		else:
			shutil.copy(x, x4)
			
# save 10 folds 
fold = 1
def save_folds(dataset, labels, path):
    global fold
    for train_index, test_index in skf.split(dataset, labels):
        print("FOLD:", fold, "TRAIN:", len(train_index), "TEST:", len(test_index))
        folder_name = 'fold'+str(fold)
        os.makedirs(os.path.join(path, folder_name))
        fold_path = path + folder_name
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        p1, p2 = create_dirs(fold_path)
        copy_imgs(X_train, X_test, y_train, y_test, p1, p2)
        fold += 1

# call
# '/home/sachin/Desktop/Guided_research/script_5_comparisons/10_fold_split/'
fold_path = args.foldpath
save_folds(dataset, labels, fold_path)