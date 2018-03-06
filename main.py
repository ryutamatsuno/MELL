

import numpy as np
from sklearn.metrics import roc_auc_score

from MELL.MELL import MELL_model
from utility import generate_train_test_data
from graph_data import GraphData


if __name__ == "__main__":


    path = 'Dataset/sample1'
    data = GraphData(path)


    test_rate = 0.2

    train_edges, test_edges = generate_train_test_data(data.L, data.N, data.directed, data.edges, test_rate)

    print("==========data description==========")
    print("total edge : ", len(data.edges))
    print("train edge : ", len(train_edges))
    print("test  edge : ", len(test_edges))
    print("====================================")

    model = MELL_model(data.L, data.N, data.directed, train_edges, 128, 4, 10, 1, 1)
    model.train(500)

    y_true = np.array(test_edges)[:, 3]
    y_predict = [ model.predict(t) for t in test_edges]

    auc = roc_auc_score(y_true, y_predict)

    print("result : " + str(auc))