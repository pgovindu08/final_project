"""
Unit tests for the k-d tree implementation (PG).
All nearest-neighbor queries are verified against a brute-force reference.
"""
import math
import random
import pytest

from kdtree import KDTree


def brute_force_nearest(points, qx, qy):
    best_id, best_dist = None, math.inf
    for node_id, x, y in points:
        d = math.sqrt((x - qx) ** 2 + (y - qy) ** 2)
        if d < best_dist:
            best_dist = d
            best_id   = node_id
    return best_id, best_dist


class TestKDTree:
    def test_single_point(self):
        tree = KDTree([(0, 1.0, 2.0)])
        nid, dist = tree.nearest(1.0, 2.0)
        assert nid == 0
        assert dist < 1e-9

    def test_exact_match(self):
        points = [(i, float(i), float(i)) for i in range(10)]
        tree   = KDTree(points)
        nid, dist = tree.nearest(5.0, 5.0)
        assert nid == 5
        assert dist < 1e-9

    def test_matches_brute_force_small(self):
        rng    = random.Random(42)
        points = [(i, rng.uniform(0, 100), rng.uniform(0, 100)) for i in range(20)]
        tree   = KDTree(points)
        for _ in range(50):
            qx, qy         = rng.uniform(0, 100), rng.uniform(0, 100)
            kd_id, kd_dist = tree.nearest(qx, qy)
            bf_id, bf_dist = brute_force_nearest(points, qx, qy)
            assert abs(kd_dist - bf_dist) < 1e-9, (
                f"query ({qx:.2f},{qy:.2f}): kd={kd_dist:.6f} bf={bf_dist:.6f}"
            )

    def test_matches_brute_force_large(self):
        rng    = random.Random(99)
        points = [(i, rng.uniform(-1000, 1000), rng.uniform(-1000, 1000))
                  for i in range(500)]
        tree   = KDTree(points)
        for _ in range(100):
            qx, qy         = rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)
            kd_id, kd_dist = tree.nearest(qx, qy)
            bf_id, bf_dist = brute_force_nearest(points, qx, qy)
            assert abs(kd_dist - bf_dist) < 1e-9

    def test_clustered_points(self):
        """All points near origin; query far away -- must still find global nearest."""
        points = [(i, float(i % 5), float(i // 5)) for i in range(25)]
        tree   = KDTree(points)
        kd_id, kd_dist = tree.nearest(100.0, 100.0)
        bf_id, bf_dist = brute_force_nearest(points, 100.0, 100.0)
        assert abs(kd_dist - bf_dist) < 1e-9

    def test_query_on_axis_boundary(self):
        """Query exactly on the splitting plane should not miss the nearest point."""
        rng    = random.Random(13)
        points = [(i, rng.uniform(0, 200), rng.uniform(0, 200)) for i in range(100)]
        tree   = KDTree(points)
        xs     = sorted(p[1] for p in points)
        qx     = xs[len(xs) // 2]
        qy     = rng.uniform(0, 200)
        kd_id, kd_dist = tree.nearest(qx, qy)
        bf_id, bf_dist = brute_force_nearest(points, qx, qy)
        assert abs(kd_dist - bf_dist) < 1e-9
