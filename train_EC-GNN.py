import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

from torch.autograd import Variable
from PIL import Image
import os, glob
import csv
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)

import seaborn as sn
imort argparse


# automate
def parse_args():
    parser = argparse.ArgumentParser(description="Create Graph data structure")
    parser.add_argument('-tr', '--train', type=str, required=True, help='pass train directory with graphs')
    parser.add_argument('-val', '--validation', type=str, required=True, help='pass validation directory with graphs')
    parser.add_argument('-cp', '--checkpoint', type=str, required=True, help='pass path to store checkpoint')
    parser.add_argument('-wt', '--save_model', type=str, required=True, help='path to store model with best validation auc')
    parser.add_argument('-sg', '--save_graphs', type=str, required=True, help='path to store training and validation graphs')

    return parser

#rcv inps from cmd
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()


# load train and val dataset in graph format from disk
class MyBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return['train_graph_ds.pt'] 

    def download(self):
        pass
    
    def process(self):
        #print('hi')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# '/home/sachin/Desktop/Guided_research/scrpit_4_r18/sparse_cosine_graph/'
train_dataset = MyBinaryDataset(root=args.train)
print('train data len', len(train_dataset))

class MyBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return['val_graph_ds.pt'] 

    def download(self):
        pass
    
    def process(self):
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# '/home/sachin/Desktop/Guided_research/scrpit_4_r18/sparse_cosine_graph/'
val_dataset = MyBinaryDataset(root=args.validation)
print('val data len', len(val_dataset))
d = val_dataset
print('number of features of each node', d.num_features)
print('# of classes: ',d.num_classes)

# count_0 = 0
# count_1 = 0

# for data in train_dataset:
# 	if data.y.item() == 0:
# 		count_0 = count_0 + 1
# 	elif data.y.item() == 1:
# 		count_1 = count_1 + 1

# print('label count class 0:', count_0)
# print('label count class_1: ', count_1)


# loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
print('train_loader len:', len(train_loader))

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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

#model = nn.DataParallel(model)
#model = model.to(device)

#print number of params
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
weights = torch.tensor([.25, .75]).to(device)
#weights = torch.tensor([.20, .80]).to(device)

# create cm
nb_classes = 2

# training    
def train(epoch):
    model.train()
    loss_all = 0.0
    train_correct = 0
    if epoch == 10:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    if epoch == 25:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.000001


    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y, weight=weights)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        train_correct += output.max(1)[1].eq(data.y).sum().item()
    return loss_all/len(train_dataset) , train_correct / len(train_dataset)


def auc(loader):
    model.eval()
    pred = []
    labl = []
    with torch.no_grad():
      for data in loader:
        #inputs, labels = inputs.to(device), labels.to(device)
        data = data.to(device)
        out = model.forward(data).max(1)[1].detach().cpu().numpy()
        labels = data.y.detach().cpu().numpy()
        pred.append(out)
        labl.append(labels)

    pred = np.hstack(pred)
    labl = np.hstack(labl)
    return roc_auc_score(labl, pred)

cm = torch.zeros(nb_classes, nb_classes)

def val(epoch):
    model.eval()
    correct = 0
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    for data in val_loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        for t, p in zip(data.y.view(-1), pred.view(-1)):
             	confusion_matrix[t.long(), p.long()] += 1
        print('confusion_matrix: ', confusion_matrix)
        per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
        print('per_class_acc: ', per_class_acc)
        correct += pred.eq(data.y).sum().item()
        if epoch == last_epoch - 1:
            for t, p in zip(data.y.view(-1), pred.view(-1)):
                cm[t.long(), p.long()] += 1
            print('cm: ', cm)
            class_acc = cm.diag()/cm.sum(1)
            print('per_class_acc at last epoch', class_acc)

    return correct / len(val_dataset)

# saving a pytorch checkpoint based on highest accuracy
# '/home/sachin/Desktop/Guided_research/scrpit_4_r18/sp_cosine_graph_results/exp.pth.tar'
# '/home/sachin/Desktop/Guided_research/scrpit_4_r18/sp_cosine_graph_results/exp.pth'
def save_checkpoint(state, is_best, filename=args.checkpoint):
    "Save check if new best is achieved"
    if is_best:
       print("=> Saving a new best")
       torch.save(state, filename) # save checkpoint
       torch.save(model, args.save_model) # save model
    else:
       print("=> Best Acc did not improve")

# save values per epoch
loss_plot = []
epoch_plot = []
val_acc_plot = []
train_acc_plot = []
val_auc_plot = []
last_epoch = 31
#best_auc = 0.0
best_acc = 0.0

for epoch in range(1, last_epoch):
    loss, train_acc = train(epoch)
    epoch_plot.append(epoch)
    loss_plot.append(loss)
    train_acc_plot.append(train_acc)
    val_acc = val(epoch)
    val_acc_plot.append(val_acc)
    val_auc = auc(val_loader)
    val_auc_plot.append(val_auc)
    is_best = bool(val_acc>best_acc)
    if val_acc > best_acc:
        best_acc = val_acc
        print('new best acc: ', best_acc)
    print('Epoch: {:02d}, Loss:{:.4f}, Train:{:.4f}, Val AUC:{:.4f},  Val: {:.4f}'.format(epoch, loss, train_acc, val_auc, val_acc))
    # save checkpoint if new best_auc
    save_checkpoint({
        'epoch': epoch ,
         'state_dict': model.state_dict(),
         'best_auc': best_acc
      }, is_best)


# save graphs
def plot_graphs(x_axis, y_axis, y_name):
  plt.plot(x_axis, y_axis, '-o')
  plt.xlabel('Epoch')
  plt.ylabel(y_name)
  plt.savefig('{}.png'.format(y_name))
  plt.clf()

plot_graphs(epoch_plot, loss_plot, args.save_graphs+'Train_Loss')
plot_graphs(epoch_plot, val_acc_plot,args.save_graphs+'val_acc')
plot_graphs(epoch_plot, train_acc_plot, args.save_graphs+'Train_acc')
plot_graphs(epoch_plot, val_auc_plot,args.save_graphs+'Val_auc')

