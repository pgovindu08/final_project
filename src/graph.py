"""
Author: Srinath Narayanan

Adjacency-list graph structure for the Knox County walking network.
Converts an OSMnx MultiDiGraph to our own representation with weights in miles.
No NetworkX objects inside this module or the algorithm code downstream.
"""

METERS_PER_MILE = 1609.344


class Graph:
    """Directed, weighted adjacency-list graph."""

    def __init__(self):
        self._adj = {}    # node_id -> list of (neighbor_id, weight_miles)
        self._nodes = set()

    def add_node(self, node_id):
        if node_id not in self._nodes:
            self._nodes.add(node_id)
            self._adj[node_id] = []

    def add_edge(self, u, v, weight_miles):
        self.add_node(u)
        self.add_node(v)
        self._adj[u].append((v, weight_miles))

    def neighbors(self, node_id):
        """Return list of (neighbor_id, weight_miles) for outgoing edges from node_id."""
        return self._adj.get(node_id, [])

    def nodes(self):
        """Return the set of all node IDs."""
        return self._nodes

    @property
    def num_nodes(self):
        return len(self._nodes)

    @property
    def num_edges(self):
        return sum(len(nbrs) for nbrs in self._adj.values())

    @classmethod
    def from_osmnx(cls, G_ox):
        """
        Convert an OSMnx MultiDiGraph (projected to meters) to our adjacency-list graph.
        Parallel edges between the same (u, v) pair are collapsed to the shortest one.
        Edge weights are stored in miles.
        """
        g = cls()
        for node_id in G_ox.nodes():
            g.add_node(node_id)

        # Collect minimum-weight edge for each directed (u, v) pair
        best = {}  # (u, v) -> weight_miles
        for u, v, data in G_ox.edges(data=True):
            weight_miles = data.get("length", 0.0) / METERS_PER_MILE
            key = (u, v)
            if key not in best or weight_miles < best[key]:
                best[key] = weight_miles

        for (u, v), w in best.items():
            g.add_edge(u, v, w)

        return g
