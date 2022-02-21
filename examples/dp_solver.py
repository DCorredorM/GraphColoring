import networkx as nx

from graph_coloring.dynamic_programming import *
import matplotlib.pyplot as plt


def draw_colored_graph(graph):
	colors = [graph.nodes[i]['color'] for i in graph.nodes]
	nx.draw(graph, node_color=colors)
	plt.show()


if __name__ == '__main__':
	g = nx.generators.random_graphs.erdos_renyi_graph(10, 0.2, seed=754)
	solve_graph_coloring(g)
	draw_colored_graph(g)







