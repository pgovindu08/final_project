"""
Author: Srinath Narayanan

Two priority-queue implementations for Dijkstra:

BinaryHeap  -- array-based min-heap with lazy deletion.
               Stale entries are left in place and skipped when popped.

IndexedHeap -- min-heap with a position map for O(log n) decrease-key.
               Maintains a dict from item -> heap index; swaps keep it consistent.
"""


class BinaryHeap:
    """
    Array-based min-heap with lazy deletion.
    push and pop are O(log n).  No heapq used.
    Lazy deletion: callers skip stale entries by comparing the popped distance
    against their own best-known distance table.
    """

    def __init__(self):
        self._heap = []   # list of (priority, item) tuples

    def push(self, priority, item):
        self._heap.append((priority, item))
        self._sift_up(len(self._heap) - 1)

    def pop(self):
        if not self._heap:
            raise IndexError("pop from empty heap")
        n = len(self._heap)
        self._swap(0, n - 1)
        priority, item = self._heap.pop()
        if self._heap:
            self._sift_down(0)
        return priority, item

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self._heap[i][0] < self._heap[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self._heap)
        while True:
            smallest = i
            left  = 2 * i + 1
            right = 2 * i + 2
            if left  < n and self._heap[left][0]  < self._heap[smallest][0]:
                smallest = left
            if right < n and self._heap[right][0] < self._heap[smallest][0]:
                smallest = right
            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break

    def _swap(self, i, j):
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

    def __len__(self):
        return len(self._heap)

    def is_empty(self):
        return len(self._heap) == 0


class IndexedHeap:
    """
    Min-heap with a position map so decrease-key runs in O(log n).
    Each item may appear at most once; use contains() before pushing.
    """

    def __init__(self):
        self._heap = []   # list of [priority, item]  (mutable for in-place update)
        self._pos  = {}   # item -> index in self._heap

    def push(self, priority, item):
        idx = len(self._heap)
        self._heap.append([priority, item])
        self._pos[item] = idx
        self._sift_up(idx)

    def pop(self):
        if not self._heap:
            raise IndexError("pop from empty heap")
        n = len(self._heap)
        self._swap(0, n - 1)
        entry = self._heap.pop()
        del self._pos[entry[1]]
        if self._heap:
            self._sift_down(0)
        return entry[0], entry[1]

    def decrease_key(self, item, new_priority):
        """Update item's priority to new_priority (must be <= current priority)."""
        i = self._pos[item]
        self._heap[i][0] = new_priority
        self._sift_up(i)

    def contains(self, item):
        return item in self._pos

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self._heap[i][0] < self._heap[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self._heap)
        while True:
            smallest = i
            left  = 2 * i + 1
            right = 2 * i + 2
            if left  < n and self._heap[left][0]  < self._heap[smallest][0]:
                smallest = left
            if right < n and self._heap[right][0] < self._heap[smallest][0]:
                smallest = right
            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break

    def _swap(self, i, j):
        item_i = self._heap[i][1]
        item_j = self._heap[j][1]
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._pos[item_i] = j
        self._pos[item_j] = i

    def __len__(self):
        return len(self._heap)

    def is_empty(self):
        return len(self._heap) == 0
