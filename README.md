# When-Convolutional-Neural-Networks-Meet-Graph-Neural-Networks-A-Case-Study-for-Glaucoma-Detection
This is my Guided Research Project done at German Research Center for AI
# Description:
This code provides a python- PyTorch, PyTorch-Geometric implemetation of a research project which uses power of two networks Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs) for Glaucoma detection in images provided within the scope of ORIGA-650 dataset. 

# Libraries Used:
Pytorch Geometric library: https://pytorch-geometric.readthedocs.io/en/latest/ and image processing libraries.

# Usage in following order:
1) python adding_conv_layers_in_r18.py
2) python create_sparse_graphs.py or python create_complete_graphs.py: store graphs with the respective format train_graph_ds.pt or val_graph_ds.pt or test_graph_ds.pt. To use edges as correlation distance or cosine similarity comment out respective functions and make some minor changes in code. 
3) python train_EC-GNN.py 
4) ensemble_models.py

Note: steps 1-4 are for ranomly split data

5) Now for Stratified 10 fold cross validation: Split data into 10 folds using split_data_with_skfcv.py and repeat steps 1-3 for each fold.
6) Perform ensemble of baseline, complete and sparse graph (correlation distance) classifiers for every fold using ensemble.py.


Note: To get a detailed description of parameters: python script.py --help

# REQUIREMENTS
1) PyTorch Geometric
2) Numpy
3) Matplotlib
4) sklearn
5) Pillow


