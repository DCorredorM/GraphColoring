from stochopti.discrete_world import mdp
import networkx as nx
from graph_coloring.base_class import Coloring
from graph_coloring.base_class import GraphColoringMDPSpace


def solve_graph_coloring(graph) -> Coloring:
	gc_space = GraphColoringMDPSpace(graph)
	gc_mdp = mdp.finiteTime(gc_space)
	policy, value = gc_mdp.solve(gc_space.epoch_states[0][0])

	coloring = Coloring(graph)
	# fill colors
	s = gc_space.epoch_states[0][0]

	for i, v in enumerate(graph.nodes):
		if i == 0:
			coloring.color_node(v, 0)
		else:
			color = policy.act((i-1, s))
			coloring.color_node(v, color)
			s = list(gc_space.transition_kernel(s, color).keys())[0]

	return coloring

