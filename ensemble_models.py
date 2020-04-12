import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('Agg')
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms, models
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
import seaborn as sn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# automate
def parse_args():
    parser = argparse.ArgumentParser(description="Create Graph data structure")
    parser.add_argument('-d_m1', '--model1_data', type=str, required=True, help='Test data path complete graphs with edges weights as cosine similarity')
    parser.add_argument('-m1wts', '--model1_wts', type=str, required=True, help='path to the complete graph model created with cosine sim')
    parser.add_argument('-d_m2', '--model2_data', type=str, required=True, help='Test data path sparse graphs with edges weights as cosine similarity')
    parser.add_argument('-m2wts', '--model2_wts', type=str, required=True, help='path to the sparse graph model created with cosine sim')
    parser.add_argument('-d_m3', '--model3_data', type=str, required=True, help='Test data path sparse graphs with edges weights as correlation distance')
    parser.add_argument('-m3wts', '--model3_wts', type=str, required=True, help='path to the sparse graph model created with correlation distance')
    parser.add_argument('-d_m4', '--model4_data', type=str, required=True, help='Test data path to the baseline')
    parser.add_argument('-m4wts', '--model4_wts', type=str, required=True, help='path to the baseline')
    parser.add_argument('-d_m5', '--model5_data', type=str, required=True, help='Test data path complete graphs with edges weights as correlation distance')
    parser.add_argument('-m5wts', '--model5_wts', type=str, required=True, help='path to the complete graph model created with correlation distance')

    return parser
#rcv inps from cmd
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()


""" model 1 """
# complete graph model created with cosine sim 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1, 6), nn.ReLU(), nn.Linear(6,  d.num_features*32))
        self.conv1 = NNConv(d.num_features, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(1, 6), nn.ReLU(), nn.Linear(6, 32*32))
        self.conv2 = NNConv(32, 32, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, d.num_classes)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        #weight = normalized_cut(data.edge_index, data.edge_attr, num_nodes=512)
        #cluster = graclus(data.edge_index, weight, data.x.size(0))
        #data.edge_attr = None
        #data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        #weight = normalized_cut(data.edge_index, data.edge_attr, num_nodes=512)
        #cluster = graclus(data.edge_index, weight, data.x.size(0))
        #x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)




# load dataset in graph format from disk
class MyBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return['test_graph_ds.pt'] 

    def download(self):
        pass
    
    def process(self):
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# '/home/sachin/Desktop/Guided_research/scrpit_4_r18/complete_graph_ds_cosine/'
test_dataset = MyBinaryDataset(root=args.model1_data)
#print('test data len', len(test_dataset))

# load model
#'/home/sachin/Desktop/Guided_research/scrpit_4_r18/exp2.2.1/exp.pth'
complete_graph_model=torch.load(args.model1_wts)

#model_sparse = torch.load('/netscratch/sharma/Guided_Research/scrpit_4_r18/edge_as_correlation_dist/edge_as_correlation_dist_sparse/exp2.3/exp.pth')
#set to eval mode
complete_graph_model.eval()

# create dataloader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#dataset ={'predict' : datasets.ImageFolder(data_dir, data_transforms['predict'])}
#dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = 1, shuffle=False, num_workers=4)}


# perform testing
outputs = list()
predictions1 = []
lbls1 = []
# get predictions
for data in test_loader:
    data = data.to(device)
    # get predictions
    output = complete_graph_model(data)
    labels = data.y.to(device)
    labels = labels.item()
    index = output.data.detach().cpu().numpy().argmax()
    predictions1.append(index)
    lbls1.append(labels)

# initialize with 2 classes
nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)

# creates confusion matrix
for t, p in zip(lbls1, predictions1):
    confusion_matrix[t, p] +=1

print('confusion_matrix: ', confusion_matrix)
per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
print('per_class_acc: ', per_class_acc)

# roc
roc = roc_auc_score(lbls1, predictions1)
print('roc cg: ', roc)

# create classification report and saving it as csv
report = classification_report(lbls1, predictions1)
print(report)

""" Model Second: sparse graph model with cosine sim """

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10,  d.num_features*16))
        self.conv1 = NNConv(d.num_features, 16, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 32*16))
        self.conv2 = NNConv(16, 32, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, d.num_classes)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        #weight = normalized_cut(data.edge_index, data.edge_attr, num_nodes=512)
        #cluster = graclus(data.edge_index, weight, data.x.size(0))
        #data.edge_attr = None
        #data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        #weight = normalized_cut(data.edge_index, data.edge_attr, num_nodes=512)
        #cluster = graclus(data.edge_index, weight, data.x.size(0))
        #x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


# load dataset in graph format from disk
class MyBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return['test_graph_ds.pt'] 

    def download(self):
        pass
    
    def process(self):
        #print('hi')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


#'/home/sachin/Desktop/Guided_research/scrpit_4_r18/sparse_cosine_graph/'
test_dataset = MyBinaryDataset(root=args.model2_data)
print('test data len', len(test_dataset))

# load model
#'/home/sachin/Desktop/Guided_research/scrpit_4_r18/sp_cosine_graph_results/exp.pth'
sp_cos_model=torch.load(args.model2_wts)

#set to eval mode
sp_cos_model.eval()

# create dataloader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#dataset ={'predict' : datasets.ImageFolder(data_dir, data_transforms['predict'])}
#dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = 1, shuffle=False, num_workers=4)}
predictions2 = []
lbls2 = []
# get predictions
for data in test_loader:
    data = data.to(device)
    # get predictions
    output = sp_cos_model(data)
    labels = data.y.to(device)
    labels = labels.item()
    index = output.data.detach().cpu().numpy().argmax()
    predictions2.append(index)
    lbls2.append(labels)

# initialize with 2 classes
nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)

# creates confusion matrix
for t, p in zip(lbls2, predictions2):
    confusion_matrix[t, p] +=1

print('confusion_matrix: ', confusion_matrix)
per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
print('per_class_acc: ', per_class_acc)

# roc
roc = roc_auc_score(lbls2, predictions2)
print('roc sg cosine: ', roc)

# create classification report and saving it as csv
report = classification_report(lbls2, predictions2)
print(report)


""" Model Third: sparse graph model with correlation distance """

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10,  d.num_features*16))
        self.conv1 = NNConv(d.num_features, 16, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 32*16))
        self.conv2 = NNConv(16, 32, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, d.num_classes)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        #weight = normalized_cut(data.edge_index, data.edge_attr, num_nodes=512)
        #cluster = graclus(data.edge_index, weight, data.x.size(0))
        #data.edge_attr = None
        #data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        #weight = normalized_cut(data.edge_index, data.edge_attr, num_nodes=512)
        #cluster = graclus(data.edge_index, weight, data.x.size(0))
        #x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

# load dataset in graph format from disk
class MyBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return['test_graph_ds.pt'] 

    def download(self):
        pass
    
    def process(self):
        #print('hi')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# '/home/sachin/Desktop/Guided_research/scrpit_4_r18/sparse_graph_dataset_based_on_correl_dist/'
test_dataset = MyBinaryDataset(root=args.model3_data)
print('test data len', len(test_dataset))

# load model
#'/home/sachin/Desktop/Guided_research/scrpit_4_r18/edge_as_correlation_dist/exp2.3/exp.pth'
model_sparse = torch.load(args.model3_wts)

#set to eval mode
model_sparse.eval()

# create dataloader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#dataset ={'predict' : datasets.ImageFolder(data_dir, data_transforms['predict'])}
#dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = 1, shuffle=False, num_workers=4)}


outputs = list()
predictions3 = []
lbls3 = []
# get predictions
for data in test_loader:
    data = data.to(device)
    # get predictions
    output = model_sparse(data)
    labels = data.y.to(device)
    labels = labels.item()
    index = output.data.detach().cpu().numpy().argmax()
    predictions3.append(index)
    lbls3.append(labels)


# initialize with 2 classes
nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)

# creates confusion matrix
for t, p in zip(lbls3, predictions3):
    confusion_matrix[t, p] +=1

print('confusion_matrix: ', confusion_matrix)
per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
print('per_class_acc: ', per_class_acc)

# roc
roc = roc_auc_score(lbls3, predictions3)
print('roc sg correlation: ', roc)

# create classification report and saving it as csv
report = classification_report(lbls3, predictions3)
print(report)




"""  Model 4: Custom Resnet Model """

# path to test data
#"/home/sachin/Desktop/Guided_research/origa_2/Test_set"
data_dir = args.model4_data

# creating my own resnet model
class myResnetModel(nn.Module):
    def __init__(self):
        super(myResnetModel, self).__init__()
        self.layer1 = child_list[0]
        self.layer2 = child_list[1]
        self.layer3 = child_list[2]
        self.layer4 = child_list[3]
        self.layer5 = child_list[4]
        self.layer6 = child_list[5]
        self.layer7 = child_list[6]
        self.layer8 = child_list[7]
        # adding my own conv layer in pretrained resnet-18 
        self.layer9 = nn.Conv2d(512, 16, 5)
        # adding fc layers for classification
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

 

    def forward(self, x):
        out1 = self.layer1(x)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = self.layer4(out1)
        out1 = self.layer5(out1)
        out1 = self.layer6(out1)
        out1 = self.layer7(out1)
        out1 = self.layer8(out1)
        out1 = self.layer9(out1)
        out1 = out1.view(-1, self.flat_features(out1))
        out1 = F.relu(self.fc1(out1))
        out1 = F.dropout(out1, training=self.training, p=0.7)
        out1 = F.log_softmax(self.fc2(out1), dim=1)
        #out1 = self.avgpool(out1)
        return out1
    
    def flat_features(self, x):
        size = x.size()[1:] # all dimensions except batch dimensions
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



# load model
# '/home/sachin/Desktop/Guided_research/scrpit_4_r18/task1_myresnet/myresnet_weights.pth'
resnet_model=torch.load(args.model4_wts)

#set to eval mode
resnet_model.eval()

# img preprocessing
data_transforms ={'predict': transforms.Compose([transforms.Resize(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.29874189944720375, 0.5893857441207402, 0.9193030837391296],
                                        std = [0.14290491468798067, 0.14531491548727832, 0.0992852656427492] 
                                        )]) }

# create dataloader
dataset ={'predict' : datasets.ImageFolder(data_dir, data_transforms['predict'])}


dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = 1, shuffle=False, num_workers=4)}
        


# initialize with 2 classes

outputs = list()
predictions4 = []
lbls4 = []
# get predictions
for inputs, labels, path in dataloader['predict']:
    path = path[0]
    img_name = path.split('/')[8]
    inputs = inputs.to(device)
    output = resnet_model(inputs)
    output = output.to(device)
    index = output.data.detach().cpu().numpy().argmax()
    labels = labels.item()
    predictions4.append(index)
    lbls4.append(labels)
    #img_names.append(img_name)

# initialize with 2 classes
nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)

# creates confusion matrix
for t, p in zip(lbls4, predictions4):
    confusion_matrix[t, p] +=1

print('confusion_matrix: ', confusion_matrix)
per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
print('per_class_acc: ', per_class_acc)

# roc
roc = roc_auc_score(lbls4, predictions4)
print('roc cr: ', roc)

# create classification report and saving it as csv
report = classification_report(lbls4, predictions4)
print(report)

""" Model 5: complete correln  """

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        nn1 = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10,  d.num_features*16))
        self.conv1 = NNConv(d.num_features, 16, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 32*16))
        self.conv2 = NNConv(16, 32, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, d.num_classes)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        #weight = normalized_cut(data.edge_index, data.edge_attr, num_nodes=512)
        #cluster = graclus(data.edge_index, weight, data.x.size(0))
        #data.edge_attr = None
        #data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        #weight = normalized_cut(data.edge_index, data.edge_attr, num_nodes=512)
        #cluster = graclus(data.edge_index, weight, data.x.size(0))
        #x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


# load dataset in graph format from disk
class MyBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return['test_graph_ds.pt'] 

    def download(self):
        pass
    
    def process(self):
        #print('hi')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# '/home/sachin/Desktop/Guided_research/scrpit_4_r18/graph_data_based_on_correlation/'
test_dataset = MyBinaryDataset(root=args.model5_data)
print('test data len', len(test_dataset))

# load model
# '/home/sachin/Desktop/Guided_research/scrpit_4_r18/edge_as_correlation_dist/exp1.6/exp.pth'
comp_corr_model=torch.load(args.model5_wts)

#set to eval mode
comp_corr_model.eval()

# create dataloader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#dataset ={'predict' : datasets.ImageFolder(data_dir, data_transforms['predict'])}
#dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = 1, shuffle=False, num_workers=4)}
predictions5 = []
lbls5 = []
# get predictions
for data in test_loader:
    data = data.to(device)
    # get predictions
    output = comp_corr_model(data)
    labels = data.y.to(device)
    labels = labels.item()
    index = output.data.detach().cpu().numpy().argmax()
    predictions5.append(index)
    lbls5.append(labels)

# initialize with 2 classes
nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)

# creates confusion matrix
for t, p in zip(lbls5, predictions5):
    confusion_matrix[t, p] +=1

print('confusion_matrix: ', confusion_matrix)
per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
print('per_class_acc: ', per_class_acc)

# roc
roc = roc_auc_score(lbls5, predictions5)
print('roc cg cosine: ', roc)

# create classification report and saving it as csv
report = classification_report(lbls5, predictions5)
print(report)



""" Ensemble Models """
# pred1: cg with cos
# pred2: sg with cos
# pred3: sg with correl
# pred4: baseline
# pred5: cg with correl

import statistics
nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)

final_pred = np.array([])

for i in range(0, len(test_dataset)):

    final_pred = np.append(final_pred, statistics.mode([predictions1[i], predictions2[i], predictions3[i],
                                                         predictions4[i]]))

# creates confusion matrix
for lbl, pred in zip(lbls1, final_pred):
    pred = int(pred)
    confusion_matrix[lbl, pred] +=1

print('confusion_matrix: ', confusion_matrix)
per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
print('per_class_acc: ', per_class_acc)

# create classification report 
report = classification_report(lbls1, final_pred)
print(report)

# roc
roc = roc_auc_score(lbls1, final_pred)
print('Roc Ensemble Models: ', roc)