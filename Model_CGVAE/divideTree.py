import collections

class DivideTree():
    def __init__(self):
        self.nodes = []
        self.edges = []

    def addNodes(self, node_feature):
        self.nodes.append(node_feature)

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
    
    def visualize(path):
        fig = plt.figure()
        ax = fig.add_subplot(111)    
        
        saddleCoords = []
        peakCoords = []
        peakElevs = []
        for i in range(nodes.shape[0]):
            tp, lati, longi, ele = nodes[i]
            if tp == 0:
                # saddle
                saddleCoords.append([lati, longi])
            else:
                # peaks
                peakCoords.append([lati, longi])
                peakElevs.append(ele)
        
        saddleCoords = np.array(saddleCoords)
        peakCoords = np.array(peakCoords)
        peakElevs = np.array(peakElevs)

        # plot ridges
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if i == j: continue
                if edges[i][j] == 1:
                    p1 = nodes[i]
                    p2 = nodes[j]
                    ax.plot([p1[1], p2[1]], [p1[2], p2[2]], color='r', linewidth=1, zorder=1)
        
        # plot peaks
        ax.scatter(peakCoords[:,0], peakCoords[:,1], marker='^', zorder=3, s=20*peakElevs/peakElevs.max(), c='white', edgecolors=(1,0.75,0,1), linewidths=1)

        # plot saddles
        ax.scatter(saddleCoords[:,0], saddleCoords[:,1], marker='o', c='white', edgecolors=(146/255, 208/255, 80/255, 1), s=6, zorder=2)
                    
        plt.savefig('test/testimg' + str(itr) + '.png')

        