"""
Author: Pranav Govindu

2-D k-d tree built from scratch.
Supports O(log n) average-case nearest-neighbor queries.
Used to snap SNAP store addresses and census tract centroids to the nearest
graph node and to compute straight-line distances for comparison.
"""

import math


class _KDNode:
    __slots__ = ("point", "node_id", "left", "right")

    def __init__(self, point, node_id, left=None, right=None):
        self.point   = point     # (x, y)
        self.node_id = node_id
        self.left    = left
        self.right   = right


class KDTree:
    """
    2-D k-d tree.  Build with KDTree(points) where points is an iterable of
    (node_id, x, y) triples.  Query with nearest(qx, qy) -> (node_id, dist).
    """

    def __init__(self, points):
        self._root = self._build(list(points), depth=0)

    def _build(self, points, depth):
        if not points:
            return None
        axis = depth % 2
        # Sort by x (axis=0) or y (axis=1); median element becomes this node
        points.sort(key=lambda p: p[1 + axis])
        mid = len(points) // 2
        node_id, x, y = points[mid]
        return _KDNode(
            point=(x, y),
            node_id=node_id,
            left =self._build(points[:mid],       depth + 1),
            right=self._build(points[mid + 1:],   depth + 1),
        )

    def nearest(self, qx, qy):
        """Return (node_id, euclidean_distance) of the nearest point to (qx, qy)."""
        best = [None, math.inf]
        self._search(self._root, qx, qy, 0, best)
        return best[0], best[1]

    def _search(self, node, qx, qy, depth, best):
        if node is None:
            return
        dx = qx - node.point[0]
        dy = qy - node.point[1]
        d  = math.sqrt(dx * dx + dy * dy)
        if d < best[1]:
            best[1] = d
            best[0] = node.node_id

        axis = depth % 2
        diff = (qx - node.point[0]) if axis == 0 else (qy - node.point[1])
        near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)

        self._search(near, qx, qy, depth + 1, best)
        # Only explore the far branch if the splitting hyperplane is within best distance
        if abs(diff) < best[1]:
            self._search(far, qx, qy, depth + 1, best)

    @classmethod
    def from_osmnx(cls, G_ox):
        """Build a k-d tree from OSMnx graph node coordinates (projected, in meters)."""
        points = [(n, data["x"], data["y"]) for n, data in G_ox.nodes(data=True)]
        return cls(points)
