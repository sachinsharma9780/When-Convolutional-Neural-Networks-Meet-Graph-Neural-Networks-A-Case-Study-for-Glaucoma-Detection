import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score
import matplotlib as mpl
mpl.use('Agg')
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image
import cv2
import argparse
# store whole data in list
#path_to_data = '/netscratch/sharma/Guided_Research/Origa650/optic_discs'
#dataset = []

# def store_data(path):
#     os.chdir(path)
#     for file in os.listdir(path):
#         np_img = cv2.imread(file)
#         dataset.append(np_img)
        
# store_data(path_to_data)


# normalize data
# X_nparray = np.array(dataset).astype(np.float64)
# X_mean = np.mean(X_nparray, axis=(0,1,2))
# X_std = np.std(X_nparray, axis=(0,1,2))

# X_nparray -= X_mean
# X_nparray /= X_std

# print(X_nparray[0])
# print(X_mean.shape)
# print('dataset mean', X_mean)
# print('dataset std', X_std)

# d_mean = [x/255 for x in X_mean]
# print('d_mean', d_mean)
# d_std = [x/255 for x in X_std]
# print('d_std', d_std)
# #d_mean [0.29874189944720375, 0.5893857441207402, 0.9193030837391296]
# #d_std [0.14290491468798067, 0.14531491548727832, 0.0992852656427492]
# # mean and std on standardised data
# print(np.mean(X_nparray, axis=(0,1,2)))
# print(np.std(X_nparray, axis=(0,1,2)))

# automate
def parse_args():
    parser = argparse.ArgumentParser(description="Create Graph data structure")
    parser.add_argument('-tr', '--train', type=str, required=True, help='pass train directory')
    parser.add_argument('-val', '--validation', type=str, required=True, help='pass validation directory')
    parser.add_argument('-cp', '--checkpoint', type=str, required=True, help='pass path to store checkpoint')
    parser.add_argument('-wt', '--save_model', type=str, required=True, help='path to store model with best validation auc')
    parser.add_argument('-sg', '--save_graphs', type=str, required=True, help='path to store training and validation graphs')

    return parser

#rcv inps from cmd
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()


# Train/validation dataset loader using subset random sampler
#train_dir = '/home/sachin/Desktop/Guided_research/origa_2/Training_set'
#val_dir = '/home/sachin/Desktop/Guided_research/origa_2/Validation_set'
train_dir = args.train
val_dir = args.validation

def load_dataset(datadir):
    data_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.29874189944720375, 0.5893857441207402, 0.9193030837391296],
                                        std = [0.14290491468798067, 0.14531491548727832, 0.0992852656427492] )
                                       ])

    data = datasets.ImageFolder(datadir,       
                    transform=data_transforms)

    loader = torch.utils.data.DataLoader(data,
                   shuffle=True, batch_size=16)

    return loader

trainloader = load_dataset(train_dir)
valloader = load_dataset(val_dir)
for data in valloader:
  print(data[0].shape)
print('number of classes:', trainloader.dataset.classes)



# Iterate DataLoader and check class balance for each batch
"""for i, (x, y) in enumerate(trainloader):
    print("batch index {}, 0/1: {}/{}".format(
        i, (y == 0).sum(), (y == 1).sum()))
    print("x.shape {}, y.shape {}".format(x.shape, y.shape))"""

# load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
model = models.resnet18(pretrained=True)

#print("resnet model", model)

# remove unnecessary layers from resnet
# converting model.children() into list and using indexing on layers
model = nn.Sequential(*list(model.children())[:-2])
print('c resnet', model)
# create list of childs
child_list = []
for child in model.children():
	child_list.append(child)

# print params of child    
#for param in child_list[0].parameters():
#    print('params', param)
#    break

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

# initialize model
myresnet_model = myResnetModel()
print('model', myresnet_model)
print('net', myresnet_model.layer9.weight.shape)

#lets try random input
#input = torch.randn(1, 3, 256, 256)
#out = myresnet_model(input)
#print('out', out)

#accesss each layer separately
# child_counter = 0
# for child in myresnet.children():
#     if child_counter == 4:
#         print('sep child 4', child[0])
#     else:
#         print(" child", child_counter, "is:")
#         print(child)

#     child_counter += 1

# freeze params of first 8 layers and redifne fully connected layer
child_counter = 0
for child in myresnet_model.children():
    if child_counter < 8:
        print("child", child_counter, "was frozen")
        for param in child.parameters():
            param.requires_grad = False
        child_counter += 1
    else:
        print("child", child_counter, "was not frozen")
        child_counter += 1

myresnet_model.to(device)
# dummy input
#input = torch.randn(2, 3, 256, 256)
#input = input.to(device)
#out = myresnet_model.forward(input)
#print('out', out.shape)
#weights = torch.tensor([.25, .75]).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(myresnet_model.parameters(), lr=0.0003, weight_decay=5e-15)
myresnet_model.to(device)


# Train a model
def train(epoch):
    myresnet_model.train()
    running_loss = 0
    train_acc = 0
    for inputs, labels, _ in trainloader:
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      logps = myresnet_model.forward(inputs)
      loss = criterion(logps, labels)
      loss.backward()
      running_loss += loss.item()
      optimizer.step()
      ps = torch.exp(logps)
      top_p, top_class = ps.topk(1, dim=1)
      equals = top_class == labels.view(*top_class.shape)
      train_acc += torch.mean(equals.type(torch.FloatTensor)).item()
    return running_loss/len(trainloader), train_acc/len(trainloader)

#  gives validation accuracy and store per class accuracy
# define number of classes
nb_classes = 2

def val():
    myresnet_model.eval()
    
    val_acc = 0
    test_loss = 0
    with torch.no_grad():
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        for inputs, labels, _ in valloader:
          inputs, labels = inputs.to(device), labels.to(device)
          logps = myresnet_model.forward(inputs)
          batch_loss = criterion(logps, labels)
          test_loss += batch_loss.item()
          ps = torch.exp(logps)
          top_p, top_class = ps.topk(1, dim=1)
          equals = top_class == labels.view(*top_class.shape)
          val_acc += torch.mean(equals.type(torch.FloatTensor)).item()
          top_p = top_p.view(-1)
          labels = labels.view(-1)
          top_class = top_class.view(-1)
          for t, p in zip(labels, top_class):
              t = np.long(t)
              p = np.long(p)
              confusion_matrix[t, p] +=1
        print('confusion_matrix: ', confusion_matrix)
        per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
        print('per_class_acc: ', per_class_acc)
        #per_class_acc = per_class_acc.detach().cpu().numpy()
        #per_class_acc = np.reshape(per_class_acc, (1, 2))
        #per_class_acc = np.append(per_class_acc, np.array(per_class_acc), axis=0)
        
    return val_acc/len(valloader), per_class_acc

# return roc_auc score

def auc(loader):
    myresnet_model.eval()
    pred = []
    labl = []
    with torch.no_grad():
      for inputs, labels, _ in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        out = myresnet_model.forward(inputs).max(1)[1].detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        pred.append(out)
        labl.append(labels)

    pred = np.hstack(pred)
    labl = np.hstack(labl)
    return roc_auc_score(labl, pred)    

# saving a pytorch checkpoint
# saving a pytorch checkpoint
# cp '/home/sachin/Desktop/Guided_research/scrpit_4_r18/my_resnet_saved_on_acc/myresnet_acc.pth.tar'
# model '/home/sachin/Desktop/Guided_research/scrpit_4_r18/my_resnet_saved_on_acc/myresnet_acc_weights.pth '
def save_checkpoint(state, is_best, filename= args.checkpoint+"myresnet_acc.pth.tar"):
    "Save check if new best is achieved"
    if is_best:
       print("=> Saving a new best")
       torch.save(state, filename) # save checkpoint
       torch.save(myresnet_model, args.save_model+"myresnet_acc_weights.pth") # save model
    else:
       print("=> Best Auc did not improve")


loss_plot = []
epoch_plot = []
val_auc_plot = []
val_acc_plot = []
train_acc_plot = []
#best_acc = 0.0
last_epoch = 31
best_auc = 0.0

for epoch in range(0, last_epoch):
    loss, train_acc = train(epoch)
    train_acc_plot.append(train_acc)
    loss_plot.append(loss)
    epoch_plot.append(epoch)
    val_auc = auc(valloader)
    val_auc_plot.append(val_auc)
    val_acc, per_class_acc = val()
    val_acc_plot.append(val_acc)
    is_best = bool(val_auc>best_auc)
    if val_auc > best_auc:
        best_auc = val_auc
        print('new best auc: ', best_auc)

   
  
    print('epoch:{:02d}, Train_loss:{:4f}, Train_acc:{:4f},Val_AUC:{:4f} ,Val_acc:{:4f}'.format(epoch, loss, train_acc, val_auc ,val_acc))
    # save checkpoint if is a new best
    save_checkpoint({
        'epoch': epoch ,
         'state_dict': myresnet_model.state_dict(),
         'best_auc': best_auc
      }, is_best)


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