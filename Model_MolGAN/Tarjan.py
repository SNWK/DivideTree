'''
tarjan({1:[2],2:[1,5],3:[4],4:[3,5],5:[6],6:[7],7:[8],8:[6,9],9:[]})
[[9], [8, 7, 6], [5], [2, 1], [4, 3]]
'''
from tarjan.tc import tc

# Python Program to detect cycle in an undirected graph 
# https://www.geeksforgeeks.org/detect-cycle-undirected-graph/
from collections import defaultdict 

class Graph: 

	def __init__(self,vertices): 
		self.V= vertices #No. of vertices 
		self.graph = defaultdict(list) # default dictionary to store graph 

	# function to add an edge to graph 
	def addEdge(self,v,w): 
		self.graph[v].append(w) #Add w to v_s list 
		self.graph[w].append(v) #Add v to w_s list 

	# A recursive function that uses visited[] and parent to detect 
	# cycle in subgraph reachable from vertex v. 
	def isCyclicUtil(self,v,visited,parent): 

		#Mark the current node as visited 
		visited[v]= True

		#Recur for all the vertices adjacent to this vertex 
		for i in self.graph[v]: 
			# If the node is not visited then recurse on it 
			if visited[i]==False : 
				if(self.isCyclicUtil(i,visited,v)): 
					return True
			# If an adjacent vertex is visited and not parent of current vertex, 
			# then there is a cycle 
			elif parent!=i: 
				return True
		
		return False
		

	#Returns true if the graph contains a cycle, else false. 
	def isCyclic(self): 
		# Mark all the vertices as not visited 
		visited =[False]*(self.V) 
		# Call the recursive helper function to detect cycle in different 
		#DFS trees 
		for i in range(self.V): 
			if visited[i] == False: #Don't recur for u if it is already visited 
				if self.isCyclicUtil(i,visited,-1): 
					return True
		return False


def calRatio(A):
    g = Graph(len(A))
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] == 1:
                g.addEdge(i, j) 

    if g.isCyclic(): 
        print("Graph contains cycle")
    else : 
        print("Graph does not contain cycle ")



def calConnectivity(A):
    G = dict()
    for i in range(len(A)):
        G[i] = []
        for j in range(len(A[i])):
            if A[i][j] == 1:
                G[i].append(j)
    groups = tc(G)
    groups:dict
    difGroups = set()
    for value in groups.values():
        difGroups.add(value)
    print("num of subgraphs: ", len(difGroups), difGroups)
    return difGroups


A = [[0,1,0,0,0],[0,0,1,0,0],[1,0,0,0,0],[0,0,0,0,1],[0,0,0,1,0]]
calConnectivity(A)
calRatio(A)