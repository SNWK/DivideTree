import collections
import matplotlib.pyplot as plt 
import numpy as np

class DivideTree():
    def __init__(self):
        self.nodes = []
        self.edges = []

    def addNodes(self, node_feature):
        if len(node_feature) == 3:
            self.nodes.append(node_feature)
        while len(node_feature[0]) != 3:
            node_feature = node_feature[0]
        self.nodes += node_feature
    
    def updateNodes(self, node_feature):
        self.nodes = []
        self.addNodes(node_feature)

    def addBond(self, node_1, node_2):
        self.edges.append((node_1, node_2))

    def removeBond(self, node_1, node_2):
        self.edges.remove((node_1, node_2))

    def checkRings(self):
        node_degree = collections.defaultdict(int)
        node_neighbors = collections.defaultdict(list)
        for e in self.edges:
            node_degree[e[0]] += 1
            node_degree[e[1]] += 1
            node_neighbors[e[0]].append(e[1])
            node_neighbors[e[1]].append(e[0])

        no_ring = True
        while True:
            no_ring = True
            one_degree_nodes = []
            for node in node_degree:
                if node_degree[node] == 1: one_degree_nodes.append(node)
                elif node_degree[node] == 2: no_ring = False
            if len(one_degree_nodes) == 0:
                break
            for node in one_degree_nodes:
                for neighbor in node_neighbors[node]:
                    node_degree[neighbor] -= 1
                node_degree[node] = 0
        # True: has ring
        return not no_ring
    
    def getSaddlesPeakInfo(self):
        """
        assume the first node is saddles
        """
        if len(self.nodes) == 0:
            return [], [], [], []
        saddleCoords = [self.nodes[0][:2]]
        peakCoords = []
        saddleElevs = [self.nodes[0][2]]
        peakElevs = []
        visited = [0]*len(self.nodes)
        issaddle = [0]*len(self.nodes)
        issaddle[0], visited[0], queue = 1, 1, [0]
        node_neighbors = collections.defaultdict(list)
        for e in self.edges:
            node_neighbors[e[0]].append(e[1])
            node_neighbors[e[1]].append(e[0])
        while len(queue) != 0:
            current_node = queue.pop()
            neighbors = node_neighbors[current_node]
            isPeak = True
            if issaddle[current_node] == 0:
                # neighbors are saddles
                isPeak = False
            for ne in neighbors:
                if visited[ne] == 0:
                    visited[ne] = 1
                    if isPeak:
                        peakCoords.append(self.nodes[ne][:2])
                        peakElevs.append(self.nodes[ne][2])
                    else:
                        issaddle[ne] = 1
                        saddleCoords.append(self.nodes[ne][:2])
                        saddleElevs.append(self.nodes[ne][2])
                    queue.append(ne)

        return saddleCoords, saddleElevs, peakCoords, peakElevs


    def visualize(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        saddleCoords, saddleElevs, peakCoords, peakElevs = self.getSaddlesPeakInfo()
        if len(saddleCoords) == 0:
            plt.savefig(path)
            return
        saddleCoords = np.array(saddleCoords)
        peakCoords = np.array(peakCoords)
        peakElevs = np.array(peakElevs)
        # plot ridges
        for e in self.edges:
            p1 = self.nodes[e[0]]
            p2 = self.nodes[e[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='r', linewidth=1, zorder=1)
        # plot peaks
        ax.scatter(peakCoords[:,0], peakCoords[:,1], marker='^', zorder=3, s=20*peakElevs/peakElevs.max(), c='white', edgecolors=(1,0.75,0,1), linewidths=1)
        # plot saddles
        ax.scatter(saddleCoords[:,0], saddleCoords[:,1], marker='o', c='white', edgecolors=(146/255, 208/255, 80/255, 1), s=6, zorder=2)
        plt.savefig(path)
    
    def remove_extra_nodes(self):
        connected = [0]*len(self.nodes)
        length = len(self.nodes)
        for edge in self.edges:
            if edge[0] in range(len(self.nodes)) and edge[1] in range(len(self.nodes)):
                connected[edge[0]] = 1
                connected[edge[1]] = 1

        mapR = dict()
        new_nodes = []
        new_edges = []

        for i in range(len(self.nodes)):
            if connected[i] == 1:
                mapR[i] = len(new_nodes)
                new_nodes.append(self.nodes[i])

        for edge in self.edges:
            if edge[0] in mapR and edge[1] in mapR:
                new_edges.append((mapR[edge[0]], mapR[edge[1]]))
        self.nodes, self.edges = new_nodes, new_edges

    def getTreeScore(self):
        # kk: TODO can be stronger
        # should be discrete
        nodes_num = len(self.nodes)
        edges_num = len(self.edges)
        score = nodes_num  - (edges_num + 1 - nodes_num)**2
        score = 1 if score > nodes_num - 4 else 0
        return score
    
    def checkCrossed(self):
        def iscross(pairData):
            A, B = pairData[0]
            C, D = pairData[1]
            AC = C - A
            AD = D - A
            BC = C - B
            BD = D - B
            CA = - AC
            CB = - BC
            DA = - AD
            DB = - BD
            return 1 if np.cross(AC,AD)*np.cross(BC,BD) < 0 and np.cross(CA,CB)*np.cross(DA,DB) < 0 else 0
        # check the last added edge
        last = self.edges[-1]
        new_edge = [np.array([self.nodes[last[0]][0], self.nodes[last[0]][1]]), 
                    np.array([self.nodes[last[1]][0], self.nodes[last[1]][1]])]
        for idx in range(len(self.edges) - 1):
            focus = self.edges[idx]
            if focus[0] in last or focus[1] in last:
                continue
            tmp_edge = [np.array([self.nodes[focus[0]][0], self.nodes[focus[0]][1]]), 
                        np.array([self.nodes[focus[1]][0], self.nodes[focus[1]][1]])]
            if iscross([new_edge, tmp_edge]):
                return True
            
        return False
