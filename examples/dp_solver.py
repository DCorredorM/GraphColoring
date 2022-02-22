from graph_coloring.dynamic_programming import *
from graph_coloring.visualizer import draw_coloring


if __name__ == '__main__':
	g = nx.generators.random_graphs.erdos_renyi_graph(10, 0.2, seed=754)
	coloring = solve_graph_coloring(g)
	draw_coloring(g, coloring)
