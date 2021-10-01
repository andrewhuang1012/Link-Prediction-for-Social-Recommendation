from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import networkx as nx
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def FourScoring(Graph, vertex1, vertex2):  # common neighbor, Jaccard's coefficient
    all_neighbor = set(G.successors(vertex1)).union(
        set(G.successors(vertex2)))  # union
    common_neighbor = set(G.successors(vertex1)) & set(
        G.successors(vertex2))  # intersection
    if(len(all_neighbor) != 0):
        Jaccard_coefficient = len(common_neighbor) / len(all_neighbor)
    else:
        Jaccard_coefficient = 0
    preferential1 = len(set(G.successors(vertex1))) + \
        len(set(G.successors(vertex2)))
    preferential2 = len(set(G.successors(vertex1))) * \
        len(set(G.successors(vertex2)))

    return len(common_neighbor), Jaccard_coefficient, preferential1, preferential2


def shortest_path_score(G, source, target):
    if(nx.has_path(G, source, target)):
        return -nx.shortest_path_length(G, source, target)
    else:
        return -7


###### Preprocessing #######
dataset = pd.read_csv("data_train_edge.csv") # Read data from csv
G = nx.DiGraph()  # Construct a directed graph
construct_set = dataset[:int(len(dataset)*0.3)]

# Construct a graph G=(V,E).
node1 = construct_set['node1'].values
node2 = construct_set['node2'].values
edge = list(zip(node1, node2))
G.add_edges_from(edge)
G.add_nodes_from(dataset['node1'])  # Add all nodes to graph
G.add_nodes_from(dataset['node2'])

record = {}  # scoring functions table

vertices = set(G.__iter__())  # All vertices on graph
# vertices.sort()
for v in vertices:
    nolink_node = set(vertices) - set(G.successors(v))
    for node in nolink_node:
        record[(v, node)] = [shortest_path_score(
            G, v, node)]  # scoring functions
        common_neighbor, Jaccard, PA1, PA2 = FourScoring(G, v, node)
        record[(v, node)].append(common_neighbor)
        record[(v, node)].append(Jaccard)
        record[(v, node)].append(PA1)
        record[(v, node)].append(PA2)

training_set = dataset[int(len(dataset)*0.3):]
newnode1 = training_set['node1'].values
newnode2 = training_set['node2'].values
newedge = list(zip(newnode1, newnode2))
G.add_edges_from(newedge)

for pair in newedge:
    record[pair].append(1)

training_data = pd.DataFrame.from_dict(record, orient='index', columns=[
                                       'Shortest', 'Common', 'Jaccard', 'PA1', 'PA2', 'label'])
training_data = training_data.fillna(0)
#############################


# Define independent variables X and dependent variables y
X = training_data.iloc[:, :5]
y = training_data.iloc[:, 5]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Load pairs for prediction
predict_set = pd.read_csv("predict.csv")
predict_node1 = predict_set['node1'].values
predict_node2 = predict_set['node2'].values

# Add all nodes to graph G
predict_edge = list(zip(predict_node1, predict_node2))
G.add_nodes_from(predict_set['node1'])
G.add_nodes_from(predict_set['node2'])

# Fill the X vector
predict_dict = {}
for nodepair in predict_edge:
    predict_dict[nodepair] = [shortest_path_score(G, nodepair[0], nodepair[1])]
    predic_cn, predict_jc, predict_PA1, predict_PA2 = FourScoring(
        G, nodepair[0], nodepair[1])
    predict_dict[nodepair].append(predic_cn)
    predict_dict[nodepair].append(predict_jc)
    predict_dict[nodepair].append(predict_PA1)
    predict_dict[nodepair].append(predict_PA2)

predict_data = pd.DataFrame.from_dict(predict_dict, orient='index', columns=[
                                      'Shortest', 'Common', 'Jaccard', 'PA1', 'PA2'])

predict_X = predict_data.iloc[:, :5]
predict_y = classifier.predict(predict_X)  # final answer to all pairs
