from graph_coloring.base_class import GraphColoringEnv
from graph_coloring.rollout import RolloutColoring
from graph_coloring.heuristics import GreedyColoring
import networkx as nx
from graph_coloring.visualizer import draw_coloring
import matplotlib.pyplot as plt
from graph_coloring.data_handler import graph_from_dimacs
import pandas as pd


def random_run(graph):
	env = GraphColoringEnv(graph)
	done = False
	state = env.reset()

	while not done:
		env.render(mode='human')
		action = env.action_space.sample()
		state, reward, done, info = env.step(action)

	env.render(mode='human')
	return state


def rollout(graph, depth=1):
	heuristic = GreedyColoring(graph)
	ro = RolloutColoring(graph, heuristic, depth)

	return ro.solve()


if __name__ == '__main__':
	# g = nx.generators.random_graphs.erdos_renyi_graph(100, 0.2, seed=754)
	all_instances = pd.read_csv('data/DimacsInstances/index.csv')

	instance = 'school1_nsh.col'
	g = graph_from_dimacs('data/DimacsInstances/Instances', instance)

	print(f'Graph {instance} with {len(g)} nodes, and {len(g.edges)} edges')

	heuristic = GreedyColoring(g)
	h_coloring = heuristic.color_graph()
	print(len(h_coloring))
	draw_coloring(g, h_coloring)
	plt.show()

	ro_coloring = rollout(g, depth=2)
	print(len(ro_coloring))
	draw_coloring(g, ro_coloring)
	plt.show()




