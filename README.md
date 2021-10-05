# Link Prediction for Social Recommendation

## Problem
In recommendation, we focus on how to recommend a suitable service or goods for users. In general, data are represented as many types. Here, we model these data and their relations as a graph. Link prediction mainly predicts the hidden connections between two users or user and goods. This method applies machine learning model to seek hidden edges on social network. We view this task as a classification work for two class. In machine learning, how to extract useful features is a main task. In our method, features are from scoring functions. 

## Definition
Given a undirected graph G=(V,E), the edge samples are defined as follows.  
Positive sample: All existing edges on graph.  
Negative sample: Edges which are still not on graph.

## Scoring functions
Jaccard coeffiecients: Here we calculate two versions JC for nodes’ successors and predecessors, respectively.

Cosine similarity: This function is to calculate the angle with given nodes’ neighbors(successors and predecessors).

Shortest path: The shortest path is meaning that there is at least one path between two nodes.

Preferential Attatchment: Sum or product of degrees of two ends.

## Files
data_train_edge.csv contains all explicit connections on the network. In prepreocessing, we only need to prepare missing edges on the network.  
If we only consider edges in data_train_edge.csv, the dataset will be an imbalanced dataset. To prevent our model from underfitting, we add missing edges to our dataset.

## Process

Step 1: Add edge pairs from data_train_edge.csv to graph G.  
Step 2: Construct positive samples.  
Step 3: Calculate scoring functions for node pairs in positive sample, labeling the positive samples.  
Step 4: Find all vertices not on graph. i.e. negative samples.  
Step 5: Calculate scoring functions for negative sample, labeling the negative sample.  
Step 6: Choose about 20,000 samples from negative samples and then combine them with positive sample to prepare the data set.  
Step 7: Use data set to fit the regression classifier.  
Step 8: Read the edges pairs from predict.csv and calculate their scoring functions.  
Step 9: Use regression classifier that already trained by data set to predict hidden edges.
Step 10: Ouput prediction results in answer.csv
