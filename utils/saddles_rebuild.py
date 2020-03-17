

def getPeaks(rootNode, isDFS=False):
    peaks = {}
    peaksEle = []
    q = list()
    q.append(rootNode)
    nodeOrder = []
    while len(q) != 0:
        node = q.pop(0)
        peaks[node.idx] = [node.x, node.y, node.ele, node.pro]
        peaksEle.append((node.idx, node.x))
        for child in node.children:
            q.append(child)
    return nodeOrder


def rebuilSaddles(rootnode):
    getPeaks 