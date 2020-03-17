import zss
import math

def weird_dist(a, b):
    dist = 0
    if a == '' or b == '':
        # remove/insert cost
        return 1000
    for i in range(4):
        # replace cost
        dist += pow(a[i] - b[i], 2)
    return math.sqrt(dist)

class WeirdNode(object):

    def __init__(self, label):
        self.my_label = label
        self.my_children = list()

    @staticmethod
    def get_children(node):
        return node.my_children

    @staticmethod
    def get_label(node):
        return node.my_label

    def addkid(self, node, before=False):
        if before:  self.my_children.insert(0, node)
        else:   self.my_children.append(node)
        return self

def buildTree(rootNode):
    root = WeirdNode(rootNode.getVecOrigin())
    q = list()
    q.append(rootNode)
    A = list()
    A.append(root)
    while len(q) != 0:
        node = q.pop(0)
        parent = A.pop(0)
        for child in node.children:
            q.append(child)
            ch = WeirdNode(child.getVecOrigin())
            parent.addkid(ch)
            A.append(ch)
    return root

def getDistance(A, B):
    return zss.simple_distance(A, B, WeirdNode.get_children, WeirdNode.get_label, weird_dist)
