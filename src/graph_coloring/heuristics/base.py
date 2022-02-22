from graph_coloring.base_class import Coloring
from abc import ABC, abstractmethod
from typing import Optional
import networkx as nx


class BaseHeuristic(ABC):
	def __init__(self, graph, **kwargs):
		self.graph = graph

	@abstractmethod
	def color_graph(self, partial_coloring: Optional = None) -> Coloring:
		...

	def reward_to_go(self, partial_coloring: Optional = None):
		coloring = self.color_graph(partial_coloring)
		return -len(coloring)


