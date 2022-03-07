from graph_coloring.base_class import Coloring
from abc import ABC, abstractmethod
from typing import Optional


class BaseHeuristic(ABC):
	"""Base class for the heuristics."""
	def __init__(self, graph, **kwargs):
		self.graph = graph
		self._cost_to_go_cache = dict()

	@abstractmethod
	def run_heuristic(self, partial_coloring: Optional = None) -> Coloring:
		"""Implement this method with the heuristic."""
		...

	def reward_to_go(self, partial_coloring: Optional = None):
		"""
		Returns the colors the more used by running the heuristic with the given partial coloring.

		Parameters
		----------
		partial_coloring: Coloring
			A partial coloring

		Returns
		-------
		int:
			The number of colors the more needed to color the whole graph based in the implemented heuristic.
		"""
		if partial_coloring is None:
			partial_coloring = Coloring(self.graph)
		_key = hash(partial_coloring)
		if _key not in self._cost_to_go_cache.keys():
			initial_colors = len(partial_coloring)
			coloring = self.run_heuristic(partial_coloring)
			cost_to_go = len(coloring) - initial_colors
			self._cost_to_go_cache[_key] = cost_to_go
		return -self._cost_to_go_cache[_key]
