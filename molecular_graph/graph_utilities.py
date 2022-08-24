import numpy as np

class Dijkstra:
    """
    written based on pseudocode from wikipedia page on Dijkstra algorithm
    https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
    """
    def __init__(self,num_nodes,adj):
        self.num_nodes = num_nodes
        self.adj = adj

    def find(self,s_idx):
        """
        Determine the shortest path between the index of source atom (s_idx) and 
        every other atoms in the molecule.
        Args:
        + atom_s (AtomNode): atom source
        Return:
        + dist (dictionary): with keys are other atoms and values are shortest distance
        """
        dist = [np.inf]*self.num_nodes
        dist[s_idx] = 0

        prev = [None]*self.num_nodes
        Q = []

        for vertex_idx in range(self.num_nodes):
            Q.append(vertex_idx)
        
        while len(Q) > 0:
            """
            u is main node
            v is neighbor nodes of main node
            """
            mini_dist = {v:dist[v] for v in Q}
            min_dist = min(mini_dist.values())
            vertices_min_d = [
                k for k,v in mini_dist.items() if v == min_dist]

            for u in vertices_min_d:
                Q.remove(u) 
                for v in self.adj[u]:

                    alt = dist[u] + 1

                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u

        return dist

if __name__ == "__main__":
    
    import sys, os

    path = os.getcwd().split("\\")
    path = "//".join(path[:-1])

    sys.path.append(path + "//molecular_graph")
    print(path + "//molecular_graph")

    from smiles import smiles2graph

    atoms, adj, bonds, bonds_feats = smiles2graph(
        "c1cc2c(cc1)c1c(ccc(c1)C#N)c1c2cccc1" )

    sp_algo = Dijkstra(len(atoms),adj)

    dist = sp_algo.find(2)
    print(dist)


