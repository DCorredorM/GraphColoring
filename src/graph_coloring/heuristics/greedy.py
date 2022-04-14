from typing import Optional

from .base import BaseHeuristic
from graph_coloring.dynamics.base_class import Coloring
from copy import copy


class GreedyColoring(BaseHeuristic):
	def __init__(self, graph, **kwargs):
		super().__init__(graph, **kwargs)

	def run_heuristic(self, partial_coloring: Optional = None) -> Coloring:
		graph = self.graph
		if partial_coloring is None:
			coloring = Coloring(graph)
		else:
			coloring = copy(partial_coloring)

		no_color = set(graph.nodes).difference(coloring.colored_nodes)

		for n in no_color:
			neighbor_colors = [coloring(n_) for n_ in graph.neighbors(n) if n_ in coloring.colored_nodes]
			color_n = 0
			while True:
				if color_n in neighbor_colors:
					color_n += 1
				else:
					coloring.color_node(n, color_n)
					break

		return coloring
