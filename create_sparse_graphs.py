import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import matplotlib as mpl
mpl.use('Agg')
from torch.autograd import Variable
from PIL import Image
import os, glob
import csv
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
import argparse
import os
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

""" 
usage eg.: python create_sparse_graph_using_correl_dist.py -l layer9 
-p0 /home/sachin/Desktop/Guided_research/origa_2/Training_set/class_0/
 -p1 /home/sachin/Desktop/Guided_research/origa_2/Training_set/class_1/ 
 -s /home/sachin/Desktop/Guided_research/scrpit_4_r18/sparse_cosine_graph/ 
 -n train_graph_ds.pt

"""

# automate
def parse_args():
    parser = argparse.ArgumentParser(description="Create Graph data structure")
    parser.add_argument('-l', '--layer', type=str, required=True, help='give layer of resnet-18')
    parser.add_argument('-p0', '--path0', type=str, required=True, help='give path of class 0')
    parser.add_argument('-p1', '--path1', type=str, required=True, help='give path of class 1')
    parser.add_argument('-s', '--saveGraph', type=str, required=True, help='path to store graph data')
    parser.add_argument('wt', 'model', type=str, required=True, help='path to the model' )
    parser.add_argument('-n', '--name', type=str, required=True, help='graph name')

    return parser
#rcv inps from cmd
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    
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

# read images from command line
#pic_one = str(input("Input first image name\n"))
#pic_two = str(input("Input second image name\n"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
#'/home/sachin/Desktop/Guided_research/scrpit_4_r18/task1_myresnet/myresnet_weights.pth'
model=torch.load(args.model)
#model = models.resnet18(pretrained=True)
#print(model)

# select desired layer which basically creates reference to the layer we want to extract from
#layer = model._modules.get('layer9')
layer = model._modules.get(args.layer)
#print(layer)

#set to eval mode so as to ensure that any dropout layers are not active during forward pass
model.eval()

# img transform
scaler = transforms.Resize((256, 256))
normalize = transforms.Normalize(mean=[0.29874189944720375, 0.5893857441207402, 0.9193030837391296],
                                        std = [0.14290491468798067, 0.14531491548727832, 0.0992852656427492])
to_tensor = transforms.ToTensor()

def get_feature_embeddings(image_name):
    # load img with Pillow library
    img = Image.open(image_name)
    # create a pytorch var with transformed image
    # unsqueeze reshape img from (3, 224, 224) to (1, 3, 224, 224) since pytorch expects 4-D input
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    t_img = t_img.to(device)
    # create vector of zeros holds feature vector
    # avg pool layer
    #my_embedding = torch.zeros(256, 16, 16)
    my_embedding = torch.zeros(16, 4, 4)
    # fn that copy o/p of layer
    """ m=module, i=grad input, o=grad ouytput """
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())

    # attach above fn to our selected layer
    h = layer.register_forward_hook(copy_data)
    # run model on transformed img
    model(t_img)
    # detach copy fn from layer
    h.remove()
    #return feature embedding vector
    return my_embedding


# testing with 1 img
#print(get_feature_embeddings('test1.jpg').shape)
#e = get_feature_embeddings('test2.jpg')
#e = e[4, :, : ].flatten()
#print('e shape', e.shape)

# holds input graph data for GNN
data_list = []

# store graph data in list
count = 0
def create_pytorch_geom_dataset(node_features, edge_index, edge_features,  label):
    global count
    count = count +1
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=label)
    print('Converting img data into graph: ', data, 'img count:', count)
    data_list.append(data)



# create edge index of graph basically adjacency matrix
def create_edge_index(ids_list, id_count):
    source_nodes = []
    target_nodes = []
    # create source node indexes
    for i in range(id_count):
            for j in range(id_count-1):
                source_nodes.append(ids_list[i].item())

    # create target node indexes
    for i in range(id_count):

        if i == 0:
            target_nodes.append(ids_list[1:])
        elif i == id_count:
            target_nodes.append(ids_list[:-1])
        else:
            lhs = ids_list[:i]
            rhs = ids_list[i+1:]
            join_list = lhs + rhs
            target_nodes.append(join_list)
    # flatten list
    target_nodes = [item for sublist in target_nodes for item in sublist]
    edge_index = torch.tensor([source_nodes,
                               target_nodes], dtype=torch.long)

    return edge_index, source_nodes, target_nodes

""" minx =-1
 maxx = 1
def calculate_cosine_sim(fv1, fv2):
     fv1 = fv1.detach().cpu().numpy()
     fv2 = fv2.detach().cpu().numpy()

     dot_product = np.dot(fv1,fv2)
     norm_a = np.linalg.norm(fv1)
     norm_b = np.linalg.norm(fv2)
     return ((dot_product/(norm_a * norm_b)-minx)/(maxx-minx))"""

def compute_cosine_sim(fv1, fv2):
    fv1 = fv1.detach().cpu().numpy()
    fv2 = fv2.detach().cpu().numpy()
    cos_lib = cosine_similarity(fv1, fv2)
    return cos_lib


def correlation_distance(feature_v1, feature_v2, node_matrix):
    feature_v1 = feature_v1.detach().cpu().numpy()
    feature_v2 = feature_v2.detach().cpu().numpy()
    # calculate pairwise distances
    p_dist = distance.pdist(node_matrix, metric='correlation')
    # convert to a square symmetric distance matrix
    sq_dist = distance.squareform(p_dist)
    sigma = np.mean(sq_dist)
    # calculate correlation distance between each pair of the feature vector
    distv = distance.cdist(feature_v1, feature_v2, metric='correlation')
    # compute simlarity measure using correlation distance and sigma
    sim = np.exp(-distv**2 / (2 * sigma**2))
    return sim

threshold = 0.5

def create_sparse_edge_index(dense_edge_index, node_matrix):
    source_nodes = []
    target_nodes = []
    source_target_pair = []
    num_edge_features = dense_edge_index.shape[1]
    for idx in range(num_edge_features):
        #print(dense_edge_index[:, idx])
        feature_vector_pair = dense_edge_index[:, idx]
        v1_idx = feature_vector_pair[0]
        v2_idx = feature_vector_pair[1]
        feature_v1 = node_matrix[v1_idx]
        feature_v1 = feature_v1.to(device)
        feature_v2 = node_matrix[v2_idx]
        feature_v2 = feature_v2.to(device)
        #feature_v1 = torch.tensor(feature_v1, dtype=torch.float)
        #feature_v2 = torch.tensor(feature_v2, dtype=torch.float)
        # compute correlation distance
        #corel_dist = correlation_distance(feature_v1.unsqueeze(0), feature_v2.unsqueeze(0), node_matrix)
        # compute cosine sim
        cosine_sim = compute_cosine_sim(feature_v1.unsqueeze(0), feature_v2.unsqueeze(0))
        print("cosine_sim shp", cosine_sim.shape)
        #corel_dist = torch.FloatTensor(corel_dist)
        if cosine_sim < threshold:
            print('removing edge connection less than threshold......', '<',threshold)
        else:
            print('Adding edge connection greater than threshold......', '>', threshold)
            source_target_pair.append(feature_vector_pair)
    for output in source_target_pair:
        source_nodes.append(output[0].item())
        target_nodes.append(output[1].item())
   
    sparse_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    return sparse_edge_index

# create edge feature using cosine similarity between two feature embeddings
def create_edge_features(edge_index, node_matrix, source_nodes, target_nodes):
    sparse_edge_index = create_sparse_edge_index(edge_index, node_matrix)
    #print('sparse_edge_index shp', sparse_edge_index.shape)

    num_edge_features = sparse_edge_index.shape[1]
    edge_features = torch.zeros([num_edge_features, 1], dtype=torch.float)
    for idx in range(num_edge_features):
        #print(edge_index[:, idx])
        feature_vector_pair = edge_index[:, idx]

        v1_idx = feature_vector_pair[0]
        v2_idx = feature_vector_pair[1]
        feature_v1 = node_matrix[v1_idx]
        feature_v1 = feature_v1.to(device)
        feature_v2 = node_matrix[v2_idx]
        feature_v2 = feature_v2.to(device)
        #feature_v1 = torch.tensor(feature_v1, dtype=torch.float)
        #feature_v2 = torch.tensor(feature_v2, dtype=torch.float)
        cosine_sim = compute_cosine_sim(feature_v1.unsqueeze(0), feature_v2.unsqueeze(0))
        cosine_sim = torch.FloatTensor(cosine_sim)
        edge_features[idx] = cosine_sim
    return edge_features, sparse_edge_index


def form_graph_node_features(feature_embeddings):
    # layer 9
    node_feature_matrix = torch.zeros([16, 16], dtype=torch.float)

    for idx in range(feature_embeddings.shape[0]):
        flatten_features = feature_embeddings[idx, :, :].flatten()
        node_feature_matrix[idx] = flatten_features
    return node_feature_matrix


def create_graph_data_structre(inp_img, label):
    feature_embeddings = get_feature_embeddings(inp_img)
    node_feature_matrix = form_graph_node_features(feature_embeddings)
    print('node feature matrix shape', node_feature_matrix.shape)
    node_index_list = torch.arange(0, node_feature_matrix.shape[0])
    node_index_list = list(node_index_list)
    edge_index, source_nodes, target_nodes = create_edge_index(node_index_list, node_feature_matrix.shape[0])
    sparse_edge_features, sparse_edge_index = create_edge_features(edge_index, node_feature_matrix, source_nodes, target_nodes)
    # flatten edge features
    sparse_edge_features = sparse_edge_features.flatten()
    create_pytorch_geom_dataset(node_feature_matrix, sparse_edge_index, sparse_edge_features, label)


# test input
#create_graph_data_structre('test1.jpg', 0)

# read data dir with respective class path
#path_class_0 = '/b_test/sharma/Guided_Research/Origa650/origa/class_0/'
path_class_0 = args.path0

for file in os.listdir(path_class_0):
        full_path = path_class_0+file
        create_graph_data_structre(full_path, 0)

#path_class_1 = '/b_test/sharma/Guided_Research/Origa650/origa/class_1/'
path_class_1 = args.path1

for file in os.listdir(path_class_1):
        full_path = path_class_1+file
        create_graph_data_structre(full_path, 1)

class MyBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return[args.name] #return ['/b_test/sharma/Guided_Research/Optic_Discs']

    def download(self):
        pass

    def process(self):
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = MyBinaryDataset(root=args.saveGraph)

