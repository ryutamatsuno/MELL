
import tensorflow as tf
import numpy as np
import random
import math
from sklearn.metrics import roc_auc_score


# random_seed = 2017
# random.seed(random_seed)
# np.random.seed(random_seed)


class MELL_model:
    """

    """

    def __init__(self, L, N, directed, edges, d, k, lamm, beta, gamma, eta = 0.075):
        """

        :param L       : num of layer
        :param N       : number of nodes
        :param directed: directed or not
        :param edges   : edge list = [layer_index, node_index, node_index, 1]
        :param d       : embedding dimension
        :param k       : number of negative samples
        :param lamm    : regularization coefficient for embedding vector
        :param beta    : regularization coefficient for variance
        :param gamma   : regularization coefficient for layer vector
        :param eta     : training rate
        """

        self.L = L
        self.N = N
        self.M = len(edges)
        self.edges = edges
        self.directed = directed

        # parameters
        self.d     = d
        self.k     = k
        self.lamm  = lamm
        self.beta  = beta
        self.gamma = gamma
        self.eta   = eta

        print('d    :', d)
        print('k    :', k)
        print('lamm :', lamm)
        print('beta :', beta)
        print('gamma:', gamma)


        self.zeroEdges = getZeroEdges(L, N, directed, edges)

        self.init_graph()

    def init_graph(self):

        d = self.d
        # #nodes
        N = self.N
        # #edges
        M = self.M
        # #layers
        L = self.L

        ########################################
        # Generate E_pos
        ########################################

        posEdges = np.array(self.edges)[:,0:3]
        if self.directed == False:
            # For undirected
            swapped = np.array(self.edges)[:,0:3]
            swapped[:,1] = posEdges[:,2]
            swapped[:,2] = posEdges[:,1]
            posEdges = np.append(posEdges,swapped,axis=0)
        self.inputPos = posEdges

        ########################################
        # Generate E_neg (whole)
        ########################################

        negEdges = np.array(self.zeroEdges)[:,0:3]
        if self.directed == False:
            # For undirected
            swapped = np.array(self.zeroEdges)[:,0:3]
            swapped[:,1] = negEdges[:,2]
            swapped[:,2] = negEdges[:,1]
            negEdges = np.append(negEdges,swapped,axis=0)
        self.inputNeg = negEdges

        ########################################
        # Struct Graph
        ########################################

        self.graph = tf.Graph()
        with self.graph.as_default():

            self.constPos = tf.constant(posEdges, dtype=tf.int32)
            self.placeNeg = tf.placeholder(tf.int32, shape=[None, 3])

            # Embedding for nodes(entities)

            # Vector as Head
            self.VH = tf.Variable(tf.random_uniform([L, N, d], -1.0, 1.0),dtype=tf.float32)

            # Vector as Tail
            if self.directed:
                self.VT = self.VH
            else:
                self.VT = tf.Variable(tf.random_uniform([L, N, d], -1.0, 1.0),dtype=tf.float32)

            # Embedding for Relation
            self.R = tf.Variable(tf.random_uniform([L, d], -1.0, 1.0),dtype=tf.float32)

            self.loss = loss(self.VH, self.VT, self.R, self.lamm, self.constPos, self.placeNeg,self.beta, self.gamma)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.eta, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)

            init = tf.global_variables_initializer()

            self.sess = tf.Session()
            self.sess.run(init)

    def train(self, max_iteration):
        """
        train part
        In the case that the stopping is dependent on the diff of the loss, the stopping part should be modified.
        :param max_iteration: number of eteration
        :return:
        """

        batch_size = int(len(self.inputPos) * self.k + 0.5)

        last_loss = 0
        for i in range(max_iteration):
            neg = getSample(self.inputNeg, batch_size)
            feed_dict = {self.placeNeg: neg}
            loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
            diff = last_loss - loss
            last_loss = loss
            if (i + 1) % 10 == 0:
                print("step %5d: loss = %15.4f, diff = %15.4f" % (i + 1, loss, diff))

        self.resVH, self.resVT, self.resR = self.sess.run((self.VH,self.VT, self.R))
        
    def save_embedding(self, dir_path):
        # save embedding results to dir_path/***.npy
        np.save(dir_path + '/VH.npy', self.resVH)
        np.save(dir_path + '/VT.npy', self.resVT)
        np.save(dir_path + '/R.npy', self.resR)

    def load_embedding(self, dir_path):
        # load embedding .npy file from dir_path/***.npy
        self.resVH = np.load(dir_path + '/VH.npy')
        self.resVT = np.load(dir_path + '/VT.npy')
        self.resR  = np.load(dir_path + '/R.npy')
        
    def __predict(self,e):
        #V, R, l, h, t
        l = e[0]
        h = e[1]
        t = e[2]

        v1 = self.resVH[l,h]
        r  = self.resR[l]
        v2 = self.resVT[l,t]

        #if self.data.directed:
        v1 = v1 + r
        return 1 / (1 + math.exp(- sum([x * y for (x,y) in zip(v1, v2)])))


    def predict(self,e):
        """

        :param e: [l h t], where l denotes the layer index, h and t denotes the head node and the tail node index, respectively.
        :return:
        """
        if self.directed:
            return  self.__predict(e)
        return  (self.__predict(e) + self.__predict([e[0],e[2],e[1]]))/2


###############################################################
# Loss functions
###############################################################

def embeddingCost(VH,VT, D, posEdges, negEdges):
    return embeddingCost_pos(VH,VT, D, posEdges) + embeddingCost_neg(VH,VT, D, negEdges)

def embeddingCost_neg(VH,VT, D, negEdges):
    # need reshape

    L,N,d = VH.get_shape().as_list() # N * N * d

    HL = tf.reshape(VH,[-1,d])
    TL = tf.reshape(VT,[-1,d])

    R = tf.gather(D,  negEdges[:, 0])
    H = tf.gather(HL, negEdges[:, 0] * N + negEdges[:, 1])
    T = tf.gather(TL, negEdges[:, 0] * N + negEdges[:, 2])

    H1 = H + R
    T1 = T

    def log1_P(v):
        return -v - tf.log(1 + tf.exp(-v))

    return - tf.reduce_sum(log1_P(tf.reduce_sum(H1 * T1, axis=1)))

def embeddingCost_pos(VH,VT, D, posEdges):

    L,N,d = VH.get_shape().as_list()

    HL = tf.reshape(VH,[-1,d])
    TL = tf.reshape(VT,[-1,d])

    R = tf.gather(D,  posEdges[:, 0])
    H = tf.gather(HL, posEdges[:, 0] * N + posEdges[:, 1])
    T = tf.gather(TL, posEdges[:, 0] * N + posEdges[:, 2])

    H1 = H + R
    T1 = T
    def logP(x):
        return -tf.log(1 + tf.exp(-x))
    return - tf.reduce_sum(logP(tf.reduce_sum(H1 * T1, axis=1)))

def loss(VH, VT, D, lamm, posEdges, negEdges,beta,gamma):

    L,N,d = VH.get_shape().as_list()

    reg_V = 1/N * lamm * (tf.reduce_sum(tf.square(VH)) + tf.reduce_sum(tf.square(VT)) )
    reg_R = gamma * tf.reduce_sum(tf.square(D))

    var_H = tf.reduce_sum(tf.square(VH - tf_stack(tf.reduce_mean(VH,axis=0),L)))
    var_T = tf.reduce_sum(tf.square(VT - tf_stack(tf.reduce_mean(VT,axis=0),L)))
    reg_E = beta * (var_H + var_T)

    cost = embeddingCost(VH, VT, D, posEdges, negEdges)
    return  cost + reg_E + reg_V + reg_R


###############################################################
# General functions
###############################################################

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


def tf_stack(x, n):
    if type(x) is list:
        shape = list(np.array(x).shape)
    else:
        shape = x.get_shape().as_list()
    t2 = tf.reshape(x, [-1])
    t3 = tf.tile(t2, [n])
    t4 = tf.reshape(t3, [n] + shape)
    return t4


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

