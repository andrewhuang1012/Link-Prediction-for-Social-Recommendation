# Link-Prediction-for-Social-Recommendation
## Definition
Given a undirected graph 𝐺=(𝑉,𝐸), the edge samples are defined as follows.  
Positive sample: All existing edges on graph.  
Negative sample: Edges which are still not on graph.

### Scoring functions:
Jaccard coeffiecients: As course contents, but here we calculate two versions JC for nodes’ successors and predecessors, respectively.

Cosine similarity: This function is to calculate the angle with given nodes’ neighbors(successors and predecessors).

Shortest path: As course contents, the shortest path is meaning that there is a connection between two nodes.

Preferential Attatchment: Sum or product of degrees of two ends.

### Process

Step 1: Add edge pairs from data_train_edge.csv to graph G.  
Step 2: Construct positive samples.  
Step 3: Calculate scoring functions for node pairs in positive sample, labeling the positive samples.  
Step 4: Find all vertices not on graph. i.e. negative samples.  
Step 5: Calculate scoring functions for negative sample, labeling the negative sample.  
Step 6: Choose about 20,000 samples from negative samples and then combine them with positive sample to prepare the data set.  
Step 7: Use data set to fit the regression classifier.  
Step 8: Read the edges pairs from predict.csv and calculate their scoring functions.  
Step 9: Use regression classifier that already trained by data set to predict hidden edges.  
