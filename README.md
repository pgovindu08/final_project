# FoodRoute -- Walking-Distance Food Access Analyzer (Knox County, TN)

**Team:** Pranav Govindu (PG) · Srinath Narayanan (SN) · Tejas Srinivasan (TS)

## Overview

FoodRoute measures how far Knox County residents must walk to reach a SNAP-authorized
supermarket.  It compares a straight-line baseline (k-d tree) against true walking
distances (multi-source Dijkstra on the OSMnx road network), classifies each census
tract using USDA food-access thresholds, and correlates results with CDC PLACES
diabetes rates.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/          # unit tests (no network required)
python run_pipeline.py           # full pipeline (downloads data on first run)
```

## Module Guide

| File | Author | Description |
|------|--------|-------------|
| `src/data_loading.py` | PG | Download Knox County walk network (OSMnx) and SNAP retailers |
| `src/graph.py` | SN | Adjacency-list graph; converts OSMnx MultiDiGraph, weights in miles |
| `src/heaps.py` | SN | `BinaryHeap` (lazy deletion) and `IndexedHeap` (true decrease-key) |
| `src/dijkstra.py` | SN | Single-source, repeated, and multi-source Dijkstra |
| `src/kdtree.py` | PG | 2-D k-d tree built from scratch; O(log n) nearest-neighbor |
| `src/snapping.py` | PG | Snap lat/lon points to nearest graph node using k-d tree |
| `src/classification.py` | PG | USDA low-access thresholds per tract |
| `src/map.py` | PG | Folium interactive map (access + disagreement + diabetes overlay) |
| `src/health.py` | TS | CDC PLACES diabetes data; Spearman correlation with access distances |
| `src/benchmarks.py` | TS | Timing experiments for all Dijkstra variants and k-d tree |
| `benchmark.py` | TS | Reusable benchmark harness (warm-up, median/min/max -> CSV) |
| `run_pipeline.py` | shared | End-to-end pipeline runner |

## Algorithm Complexity

### Repeated Single-Source Dijkstra (baseline)

Run one Dijkstra from each of the **N** store nodes.  With a binary heap:

```
O(N * (V + E) log V)
```

Each pass visits every edge once and does O(log V) heap operations.  Total cost
scales linearly with the number of stores -- tractable for Knox County (~10 stores)
but slower than the multi-source approach.

### Multi-Source Dijkstra (central algorithm)

Insert **all** store nodes at distance 0, then run a single Dijkstra pass:

```
O((V + E) log V)
```

The number of stores does not appear in the complexity -- the wavefront visits every
edge at most once regardless of how many stores there are.  Every reachable
intersection is labeled with the nearest store ID and walking distance in one pass.

**Label propagation:** each node inherits the store label of whatever path relaxed it
to its current shortest distance.  Ties (two stores equidistant to a node) are broken
by heap pop order, giving a deterministic but arbitrary assignment.

### Lazy Deletion vs. True Decrease-Key

| Heap | push | pop | decrease-key | space |
|------|------|-----|--------------|-------|
| `BinaryHeap` (lazy) | O(log n) | O(log n) amortised | N/A -- push duplicate | O(E) worst case |
| `IndexedHeap` | O(log n) | O(log n) | O(log n) | O(V) |

With lazy deletion, stale entries accumulate (up to one per relaxation), so the heap
can grow to O(E) entries.  Checking staleness is cheap: compare the popped distance
against `dist[u]`.

The indexed heap pays extra memory for a position map (`item -> heap index`) but
keeps the heap at O(V) entries by updating in place.  On dense graphs with many
relaxations, the indexed heap can be faster in practice; on sparse graphs, the
difference is small.  Tejas's benchmark harness (`src/benchmarks.py`) measures
both and writes results to CSV for the complexity plots.