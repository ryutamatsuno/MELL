

import os

class GraphData:


    def __init__(self,path):

        data_set_name = os.path.basename(path)

        # load edges
        f = open(path + "/" + data_set_name + '_multiplex.edges')
        lines = f.readlines()
        f.close()
        edges = [[e[0], e[1], e[2], 1] for e in
                 [[int(float(c)) - 1 for c in line.split(' ')] for line in lines if len(line) > 6]]

        # load info
        info_path = path + "/" + data_set_name + '_info.txt'
        f = open(info_path)
        lines = f.readlines()
        f.close()

        info = {}
        for l in lines:
            l = l.replace('\n', '')
            ls = l.split(':')
            key = ls[0]
            val = ls[1]
            info[key] = val

        # property
        self.L = int(info['L'])
        self.N = int(info['N'])
        self.M = int(info['M'])
        self.directed = (info['directed'] == 'True')
        self.edges = edges
        #return L, N, directed, edges




