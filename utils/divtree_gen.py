import math

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 
    return c * r

class Graph:
    def __init__(self):
        self.edges = []
        self.vertices = set()
        
    def add_edge(self, v, w, weight):
        self.vertices.add(v)
        self.vertices.add(w)
        self.edges.append((v,w,weight))
        
    def E(self):
        return self.edges
     
    def V(self):
        return self.vertices


class UnionFind:
    def __init__(self, vertices):
        self.size = len(vertices)
        self.components = {}
        self.tree_sizes = {}
        for v in vertices:
            self.components[v] = v
            self.tree_sizes[v] = 1
        
    def connect(self, a, b):
        a_root = self.root(a)
        b_root = self.root(b)
        if(a_root is not b_root): 
            if(self.tree_sizes[a_root] < self.tree_sizes[b_root]):
                self.components[a_root] = b_root
                self.tree_sizes[b_root] += self.tree_sizes[a_root]
            else: 
                self.components[b_root] = a_root
                self.tree_sizes[a_root] += self.tree_sizes[b_root]
       
    def connected(self, a, b):
        return self.root(a) is self.root(b)
    
    def root(self, a):
        while a is not self.components[a]:
            # path compression optimization
            self.components[a] = self.components[self.components[a]] 
            a = self.components[a]
        return a

class Treenode:
    def __init__(self, idx, x, y, ele, pro, par):
        self.idx= idx
        self.x = x
        self.y = y
        self.ele = ele
        self.pro = pro
        self.parent = par
        self.children = []
    
    def getVec(self):
        if self.parent == None:
            return (0,0,0,0,0)
        else:
            dist = haversine(self.parent.x, self.parent.y, self.x, self.y)
            arc = math.atan((self.y - self.parent.y)/(self.x - self.parent.x)) / 2*math.pi
            return (0, self.x - self.parent.x, self.y - self.parent.y, self.ele - self.parent.ele, self.pro - self.parent.pro)

def byWeight(edge):
    return edge[2]

# kruskal's algorithm
def minimum_spanning_tree(tree):
    result = []
    uf = UnionFind(tree.V())
    edges = sorted(tree.E(), key=byWeight)
    while edges:
        minor_edge = edges[0]
        edges.remove(minor_edge)
        v = minor_edge[0]
        w = minor_edge[1]
        if not uf.connected(v, w):
            result.append(minor_edge)
            uf.connect(v, w)
    # return all connected edges
    return result

def buildTree(edges):
    return

def getTreeNodeByIdx(root, index):
    if root.idx == index:
        return root
    queue = []
    queue.append(root)
    while queue:
        node = queue.pop()
        for i in node.children:
            if i.idx == index:
                return i
            else:
                queue.append(i)
    print("Error: not found", index)


def genDivideTree(peaks):
    # for RNN
    vertices = peaks.index
    gsample = Graph()
    pairs = set()
    for v in vertices:
        for w in vertices:
            if v != w and (v, w) not in pairs and (w, v) not in pairs:
                pairs.add((v, w))
                lat1 = peaks['latitude'].loc[v]
                lon1 = peaks['longitude'].loc[v]
                lat2 = peaks['latitude'].loc[w]
                lon2 = peaks['longitude'].loc[w]
                dist = haversine(lon1, lat1, lon2, lat2)
                gsample.add_edge(v, w, dist)
    treesample = minimum_spanning_tree(gsample)

    # For test, draw the tree
    # drawTree(peaks, treesample)
    
    rootidx = peaks['elevation in feet'].idxmax()
    rootNode = Treenode(rootidx, peaks['longitude'].loc[rootidx], 
                        peaks['latitude'].loc[rootidx], peaks['elevation in feet'].loc[rootidx], 
                        peaks['prominence in feet'].loc[rootidx], None)
    
    edges = {}
    for edge in treesample:
        v, w, _ = edge
        if v not in edges.keys():
            edges[v] = []
        if w not in edges.keys():
            edges[w] = []
        edges[v].append(w)
        edges[w].append(v)
    
    queue = []
    queue.append(rootidx)
    while queue:
        nodeidx = queue.pop()
        node = getTreeNodeByIdx(rootNode, nodeidx)
        # print("parent",node.idx)
        for childrenidx in edges[nodeidx]:
            # print("parent",node.idx, "child", childrenidx)
            childnode = Treenode(childrenidx, peaks['longitude'].loc[childrenidx], 
                        peaks['latitude'].loc[childrenidx], peaks['elevation in feet'].loc[childrenidx], 
                        peaks['prominence in feet'].loc[childrenidx], node)
            node.children.append(childnode)
            edges[childrenidx].remove(nodeidx)
            queue.append(childrenidx)

    return rootNode


def getTreeHMC(peaks):
    gsample = Graph()
    pairs = set()
    for v in range(len(peaks)):
        for w in range(len(peaks)):
            if v != w and (v, w) not in pairs and (w, v) not in pairs:
                pairs.add((v, w))
                lat1 = peaks[v][1]
                lon1 = peaks[v][0]
                lat2 = peaks[w][1]
                lon2 = peaks[w][0]
                dist = haversine(lon1, lat1, lon2, lat2)
                gsample.add_edge(v, w, dist)
    treesample = minimum_spanning_tree(gsample)
    result = []
    for r in treesample:
        v, w, _ = r
        result.append([peaks[v], peaks[w]])
    return result
