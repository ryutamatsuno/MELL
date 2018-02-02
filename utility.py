

import numpy as np


def getZeroEdges(L, N, directed, edges):
    A = adjacencyMatrix(L, N, directed, edges)
    zeroEdges = []
    for l in range(L):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if directed == False:
                    if i < j:
                        continue
                if A[l,i,j] == 0:
                    zeroEdges.append([l, i, j, 0])
    return zeroEdges

def adjacencyMatrix(L, N, directed,edges):
    A = np.zeros((L, N, N))
    for e in edges:
        A[e[0]][e[1]][e[2]] = e[3]
        if directed == False:
            A[e[0]][e[2]][e[1]] = e[3]
    return A


def getSample(data, sample_size):
    whole = len(data)
    if whole == sample_size:
        return data
    elif whole < sample_size:
        sample1 = getSample(data,whole)
        sample2 = getSample(data,sample_size - whole)
        print(type(sample1))
        return sample1 + sample2
    permutation = np.random.permutation(whole)
    permutation = permutation[0:sample_size]
    sampled = np.array(data)[permutation]
    return sampled.tolist()



def add_negative_tests_uniform(L, N, directed, edges, tests):
    zeroEdgeList  = getZeroEdges(L, N, directed, edges)
    num_positive = len(tests)
    negative_sample = getSample(zeroEdgeList,num_positive)
    all_test = negative_sample + tests
    return all_test


def generate_train_test_data(L, N, directed, edges, test_rate):
    trains = []
    tests = []

    npedges = np.array(edges)
    edges_on_layer = [(npedges[npedges[:, 0] == i]).tolist() for i in range(L)]

    for l in range(L):
        le = edges_on_layer[l]
        lm = len(le)
        test_indexes = getSample(list(range(lm)), int(lm * test_rate + 0.5))
        for i,e in enumerate(le):
            if i in test_indexes:
                tests.append(e)
            else:
                trains.append(e)

    tests = add_negative_tests_uniform(L, N, directed, edges, tests)
    return trains, tests






