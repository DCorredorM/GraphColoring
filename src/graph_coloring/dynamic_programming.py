from stochopti.discrete_world import mdp, space
import networkx as nx


class GraphColoringSpace(space.finiteTimeSpace):
	def __init__(self, graph: nx.Graph):
		self.graph = graph
		actions = range(len(graph))
		self.epoch_states = self.get_states()
		super().__init__(sum(self.epoch_states, []), actions, len(graph))

		self.T = len(graph)

	def build_admisible_actions(self):
		def adm_A(s: frozenset):
			actions = list()
			t = sum(len(c) for c in s)
			if t >= len(list(self.graph.nodes)):
				return [None]
			v_t = list(self.graph.nodes)[t]
			for i, c in enumerate(s):
				if all((v_t, v) not in self.graph.edges for v in c):
					actions.append(i)

			actions.append(len(s))

			return frozenset(actions)

		return adm_A

	def build_kernel(self):
		def Q(state, action):
			t = sum(len(c) for c in state)
			if t >= len(list(self.graph.nodes)):
				return {state: 1}
			v_t = list(self.graph.nodes)[t]

			if action < len(state):
				s_ = list(state)
				s_[action] = state[action].union([v_t])
				s_ = tuple(s_)
			else:
				s_ = state + (frozenset([v_t]),)
			return {s_: 1}

		return Q

	def reward(self, state, action=None, time=None):
		if action is None:
			return -1
		s_ = list(self.Q(state, action).keys())[0]
		return - (len(s_) - len(state))

	def get_states(self):
		S = [[(frozenset([list(self.graph.nodes)[0]]),)]]

		def succ(s):
			succ_ = list()
			t = sum(len(c) for c in s)
			v_t = list(self.graph.nodes)[t]
			for i, c in enumerate(s):
				if all((v_t, v) not in self.graph.edges for v in c):
					s_ = list(s)
					s_[i] = c.union([v_t])
					succ_.append(tuple(s_))

			succ_.append(s + (frozenset([v_t]),))

			return succ_

		for i in range(1, len(self.graph)):
			states = [succ(s) for s in S[i-1]]
			S.append(sum(states, []))
		
		return S


def solve_graph_coloring(graph):
	gc_space = GraphColoringSpace(graph)
	gc_mdp = mdp.finiteTime(gc_space)
	policy, value = gc_mdp.solve(gc_space.epoch_states[0][0])

	# fill colors
	s = gc_space.epoch_states[0][0]

	for i, v in enumerate(graph.nodes):
		if i == 0:
			graph.nodes[v]['color'] = 0
		else:
			graph.nodes[v]['color'] = policy.act((i-1, s))
			s = list(gc_space.Q(s, graph.nodes[v]['color']).keys())[0]

	return abs(value)

