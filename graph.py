from math import *
import shapely as sh
from shapely.geometry import *
from shapely.affinity import *
from shapely.prepared import prep
import graphviz as gv
import random as rand
import matplotlib.pyplot as plt
import re


class Graph:
    def __init__(self):
        self.vertices = dict() # from a label to a point
        self.adj = dict() # a dict of sets

    def add_vertex(self, label, point=None):
        self.vertices[label] = point

    def add_edge(self, u, v):
        # add edge to vertex list if it's not already present
        if not self.vertices.get(u):
            self.add_vertex(u)
        if not self.vertices.get(v):
            self.add_vertex(v)

        # add edge to the adjacency list
        if self.adj.get(u):
            self.adj[u].add(v)
        else:
            self.adj[u] = set([v])

        if self.adj.get(v):
            self.adj[v].add(u)
        else:
            self.adj[v] = set([u])

    def remove_edge(self, u, v):
        self.adj[u].discard(v)
        self.adj[v].discard(u)

    def to_graphviz(self):
        out = 'graph G {\n'
        for v in self.vertices:
            if not self.adj.get(v):
                out += ' "{}";\n'.format(v)
            else:
                for w in self.adj[v]:
                    if (v <= w):
                        out += ' "{}" -- "{}";\n'.format(v, w)
        out += '}'
        return out
