from graph_coloring.base_class import GraphColoringEnv, Coloring
from graph_coloring.heuristics.base import BaseHeuristic


Action = int


class RolloutColoring:
	def __init__(self, graph, heuristic: BaseHeuristic, depth, **kwargs):
		self.graph = graph

		self.env = GraphColoringEnv(graph)
		self.heuristic = heuristic

		self.max_depth = depth

		self._temp_path = []
		self._temp_reward = float('-inf')

	def update_temporary_search(self, temp_path, temp_reward):
		cur_state = temp_path[-1]
		heuristic_reward = self.heuristic.reward_to_go(cur_state)
		if temp_reward + heuristic_reward > self._temp_reward:
			self._temp_reward = temp_reward + heuristic_reward
			self._temp_path = temp_path

	def check_pruning_strategies(self, temp_path, temp_reward):
		return True

	def roll_out_search(self, rollout_path, rollout_reward, depth):
		if depth >= self.max_depth:
			self.update_temporary_search(rollout_path, rollout_reward)
		else:
			for action in self.env.action_space.feasible_actions(rollout_path[-1]):
				next_state = self.env.simulate_transition_state(action, rollout_path[-1])
				instant_reward = self.env.dynamics.reward_(rollout_path[-1], next_state, action)
				new_path = rollout_path + [action, next_state]
				new_reward = rollout_reward + instant_reward

				if self.check_pruning_strategies(new_path, new_reward):
					self.roll_out_search(rollout_path=new_path, rollout_reward=new_reward, depth=depth + 1)

	def roll_out(self, state) -> Action:
		self._temp_path = [state]
		self._temp_reward = float('-inf')

		self.roll_out_search(self._temp_path, 0, 0)

		return self._temp_path[1]

	def solve(self):
		initial_state = self.env.reset()
		path = [initial_state]
		total_cost = 0

		coloring = Coloring(self.graph)
		coloring.color_node(list(self.graph.nodes)[0], 0)
		done = False
		node_index = 1
		
		while not done:
			action = self.roll_out(path[-1])
			next_state, cost, done, info = self.env.step(action)
			path += [action, next_state]
			total_cost += cost

			coloring.color_node(list(self.graph.nodes)[node_index], action)
			node_index += 1
		return coloring







