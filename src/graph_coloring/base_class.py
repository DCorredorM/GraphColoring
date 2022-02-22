from copy import copy
from typing import Any, List
from itertools import product
import networkx as nx
from stochopti.discrete_world import space
from gym import Space, Env
from gym.spaces import Discrete
import numpy as np

from graph_coloring.visualizer import draw_coloring, Visualizer


class GraphColoringMDPSpace(space.finiteTimeSpace):
	def __init__(self, graph: nx.Graph, build_states=True):
		self.graph = graph
		actions = range(len(graph))
		if build_states:
			self.epoch_states = self.get_states()
		else:
			self.epoch_states = [[]]
		super().__init__(sum(self.epoch_states, []), actions, len(graph))

		self.T = len(graph)

	def build_admisible_actions(self):
		def adm_A(s: Coloring):
			t = sum(len(c) for c in s)
			if t >= len(list(self.graph.nodes)):
				return [None]
			v_t = list(self.graph.nodes)[t]

			return s.feasible_colors(v_t)

		return adm_A

	def build_kernel(self):
		def Q(state: Coloring, action):
			t = len(state.colored_nodes)
			if t >= len(list(self.graph.nodes)):
				return {state: 1}
			v_t = list(self.graph.nodes)[t]

			s_ = copy(state)
			s_.color_node(v_t, action)

			return {s_: 1}

		return Q

	@staticmethod
	def reward_(state, next_state=None, action=None):
		if action is None:
			return -1
		s_ = next_state
		return - (len(s_) - len(state))

	def reward(self, state, action=None, time=None):
		if action is None:
			return -1
		s_ = list(self.transition_kernel(state, action).keys())[0]
		return self.reward_(state, s_, action)

	def get_states(self):
		initial_sate = Coloring(self.graph)
		initial_sate.color_node(list(self.graph.nodes)[0], 0)
		S = [[initial_sate]]

		def succ(s: Coloring):
			succ_ = list()
			t = len(s.colored_nodes)
			v_t = list(self.graph.nodes)[t]

			for i, c in enumerate(s):
				s_ = copy(s)
				if s_.color_node(v_t, i, strict=True):
					succ_.append(s_)

			s_ = copy(s)
			s_.color_node(v_t, len(s_))
			succ_.append(s_)

			return succ_

		for i in range(1, len(self.graph)):
			states = [succ(s) for s in S[i - 1]]
			S.append(sum(states, []))

		return S


class GraphColoringActionSpace(Discrete):
	def __init__(self, n, env: 'GraphColoringEnv'):
		super().__init__(n)
		self.env = env

	def sample(self):
		cur_state = self.env.observation_space.current_state
		feasible_actions = self.env.dynamics.admisible_actions(cur_state)

		return self.np_random.choice(feasible_actions)

	def feasible_actions(self, state=None):
		if state is None:
			cur_state = self.env.observation_space.current_state
		else:
			cur_state = state
		return self.env.dynamics.admisible_actions(cur_state)


class GraphColoringStateSpace(Space):
	def __init__(self, graph: nx.Graph):
		super().__init__()
		self.graph = graph
		self.current_state = Coloring(graph)

	def reset_observation_space(self):
		c = Coloring(self.graph)
		initial_node = list(self.graph.nodes)[0]
		c.color_node(initial_node, 0)
		self.current_state = c

	def sample(self):
		epoch = np.random.randint(0, len(self.graph))
		coloring = Coloring(self.graph)

		for i in range(epoch):
			colored = False
			node = list(self.graph.nodes)[i]
			while not colored:
				color = np.random.randint(0, len(coloring))
				colored = coloring.color_node(node, color, strict=True)

	def contains(self, x):
		if isinstance(x, Coloring):
			return x.conflicting_pairs() == 0
		else:
			return False


class GraphColoringEnv(Env):

	def __init__(self, graph: nx.Graph):
		self.graph = graph
		self.observation_space = GraphColoringStateSpace(graph)
		self.action_space = GraphColoringActionSpace(len(self.graph), self)
		self.dynamics = GraphColoringMDPSpace(self.graph, build_states=False)

		self.visualizer = None
		self._done = False

	def simulate_transition_state(self, action, state=None):
		if state is None:
			cur_state = self.observation_space.current_state
		else:
			cur_state = state
		distribution = self.dynamics.transition_kernel(cur_state, action)
		values = list(distribution.keys())
		probabilities = list(distribution.values())
		index = np.random.choice(a=range(len(values)), p=probabilities)

		return values[index]

	def step(self, action):
		cur_state = self.observation_space.current_state
		next_state: Coloring = self.simulate_transition_state(action)
		reward = self.dynamics.reward_(cur_state, next_state, action)

		info = {'state': next_state, 'colored_nodes': next_state.colored_nodes}
		if next_state.is_coloring():
			done = True
		else:
			done = False
		info['found_coloring'] = done

		self.observation_space.current_state = next_state
		self._done = done
		return next_state, reward, done, info

	def reset(self):
		self.observation_space.reset_observation_space()
		self._done = False
		return self.observation_space.current_state

	def render(self, mode='human'):
		# just raise an exception
		if mode == 'ansi':
			print(self.observation_space.current_state)
		elif mode == 'human':
			if self.visualizer is None:
				self.visualizer = Visualizer(graph=self.graph)

			self.visualizer.render(self.observation_space.current_state, final=self._done)
		else:
			super(GraphColoringEnv, self).render(mode=mode)


class Coloring(list):
	def __init__(self, graph: nx.Graph, *args, **kwarg):
		super().__init__(*args, **kwarg)
		self._coloring = dict()
		self.graph: nx.Graph = graph
		self.colored_nodes = set()

	def color_node(self, node: Any, color: int, strict=True):
		if strict:
			return self._color_node_strict(node, color)
		else:
			return self._color_node_soft(node, color)

	def __call__(self, node):
		return self._coloring.get(node, None)

	def __copy__(self):
		cls = self.__class__
		obj = cls(self.graph)
		for n, c in self._coloring.items():
			obj.color_node(n, c)

		return obj

	def __hash__(self):
		return hash(f'{self.graph}-{self}')

	def get_color(self, node):
		return self(node)

	def conflicting_pairs(self):
		c_pairs = []
		for same_color in self:
			for (i, j) in product(same_color, same_color):
				if (i, j) in self.graph.edges:
					c_pairs.append((i, j))
		return c_pairs

	def is_coloring(self):
		is_partition = set(self._coloring.keys()) == set(self.graph.nodes)
		zero_conflict = len(self.conflicting_pairs()) == 0

		return is_partition and zero_conflict

	def number_of_colors(self):
		if self.is_coloring():
			return len(self)
		else:
			return float('inf')

	def check_if_node_can_be_colored(self, node: Any, color: int):
		flag = True
		if color >= len(self):
			return flag

		for n_ in self[color]:
			if (n_, node) in self.graph.edges:
				flag = False
				break
		return flag

	def feasible_colors(self, node: Any) -> List[int]:
		feasible_colors = []

		for i, _ in enumerate(self):
			if self.check_if_node_can_be_colored(node, i):
				feasible_colors.append(i)

		feasible_colors.append(len(self))
		return feasible_colors

	def _color_node_strict(self, node: Any, color: int):
		if self.check_if_node_can_be_colored(node, color):
			return self._color_node_soft(node, color)
		else:
			return False

	def _color_node_soft(self, node: Any, color: int):
		cur_color = self._coloring.get(node, None)
		if cur_color is not None:
			self[cur_color].remove(node)
		else:
			self.colored_nodes.add(node)

		self._coloring[node] = color
		if len(self) <= color:
			self.append({node})
		else:
			self[color].add(node)

		return True
