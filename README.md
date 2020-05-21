# When-Convolutional-Neural-Networks-Meet-Graph-Neural-Networks-A-Case-Study-for-Glaucoma-Detection
This is my Guided Research Project done at German Research Center for AI

# Abstract
Convolutional Neural Networks (CNNs) are more successful in problems where underlying geometry of the data lies in euclidean space or follows a regular grid like structure. Common examples of such problems include image classiﬁcation, audio analysis, and natural language processing, where CNNs are quite powerful in extracting features that are useful in decision making. However, many real-world problems may follow non-euclidean geometry such as social networks, medical diagnosis, and genetics etc., where semantic relational information is also vital. In medical image classiﬁcation tasks semantic associations among discriminating features drawn from the images can be useful for classiﬁcation since many features may appear together in images belonging to certain class or disease. This relational information can be modelled by graphs and processed through Graph Neural Networks (GNNs), which have shown considerable strength in handling graph data structures. In this paper we propose a deep learning based framework that combines the strengths of both CNNs, as powerful feature extractors, and GNNs, as capable of handling relational information, for detection of glaucoma from retinal fundus images using a large publicly available ORIGA dataset. The framework consists of three modules: Feature Extractor, Graph Constructor and Graph Classiﬁer. We performed detailed experiments exploring diﬀerent graph construction approaches along with diﬀerent similarity measures. We also combine CNNs and GCNs using ensemble and achieved improved classiﬁcation performance. Our results with GCN are promising and competitive to state-of-the-art results with 0.79 precision, 0.76 recall and 0.77 F-1 score on ORIGA with an ensemble of CNN and GCNs.

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


