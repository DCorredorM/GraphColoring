from graph_coloring.rollout import RolloutColoring
from graph_coloring.heuristics import GreedyColoring
import networkx as nx
from utilities.counters import Timer
from time import sleep
from graph_coloring.data_handler import *
import pandas as pd


def rollout():
	all_instances = pd.read_csv('data/DimacsInstances/index.csv')
	#
	instance = 'mulsol.i.5.col'
	graph = graph_from_dimacs('data/DimacsInstances/Instances', instance)
	# graph = nx.generators.random_graphs.erdos_renyi_graph(10, 0.2, seed=754)

	# print(f'Graph {instance} with {len(g)} nodes, and {len(g.edges)} edges')
	t = Timer('Heuristic', verbose=True)
	t.start()
	heuristic = GreedyColoring(graph)
	h_coloring = heuristic.run_heuristic()
	t.stop()
	print(len(h_coloring))
	# draw_coloring(g, h_coloring)
	# plt.show()
	sleep(0.1)

	t = Timer('Rollout', verbose=True)
	heuristic = GreedyColoring(graph)
	ro = RolloutColoring(graph, heuristic, depth=2)
	t.start()
	ro_coloring = ro.solve()
	t.stop()
	print(len(ro_coloring))
	print(f'Bound pruning: {ro.bound_counter}')
	print(f'Heuristic calls: {ro.heuristics_call}')


	# draw_coloring(g, ro_coloring)
	# plt.show()


def main():
	rollout()


if __name__ == '__main__':
	main()




