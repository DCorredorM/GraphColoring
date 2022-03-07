from graph_coloring.base_class import GraphColoringEnv, Coloring
from graph_coloring.heuristics.base import BaseHeuristic
import logging

from utilities.counters import TallyCounter

logging.basicConfig(level=logging.DEBUG)

Action = int


class RolloutColoring:
	"""
	An Implementation of the rollout policy for the Graoph coloring problem.

	Attributes
	----------
	graph: nx.Graph
		The graph to be colored
	heuristic: BaseHeuristic
		The heuristic to use.
	max_depth: int
		The maximum depth to use in the rollout online exploration.
	total_reward: int
		The estimated chromatic number for the given graph
	"""
	def __init__(self, graph, heuristic: BaseHeuristic, depth, **kwargs):
		self.graph = graph

		self.env = GraphColoringEnv(graph)
		self.heuristic = heuristic

		self.max_depth = depth

		self._temp_path = []
		# Local bound	
		self._temp_reward = self.heuristic.reward_to_go()

		# Global bound
		self.total_reward = self._temp_reward

		self.logger = logging.getLogger(__name__)

		_, self.max_degree = max(self.graph.degree(self.graph.nodes), key=lambda x: x[1])

		self.bound_counter = TallyCounter(name='Bound pruning')
		self.heuristics_call = TallyCounter(name='heuristic call')

	def update_temporary_search(self, temp_path, temp_reward):
		"""
		Updates the online exploration incumbent path and reward.

		Parameters
		----------
		temp_path:
			The current temporary path. A sequence of state and action paris that indicates the path used.
		temp_reward:
			Temporary number of colors used.
		"""
		cur_state = temp_path[-1]
		heuristic_reward = self.heuristic.reward_to_go(cur_state)
		self.heuristics_call.count()
		if temp_reward + heuristic_reward > self._temp_reward:
			self._temp_reward = temp_reward + heuristic_reward
			self._temp_path = temp_path

		if self._temp_reward > self.total_reward:
			self.total_reward = self._temp_reward

	def check_pruning_strategies(self, temp_path, temp_reward):
		"""
		Checks the pruning strategies for a given partial path.

		Parameters
		----------
		temp_path:
			The current temporary path. A sequence of state and action paris that indicates the path used.
		temp_reward:
			Temporary number of colors used.

		Returns
		-------
		bool
			True if the path should be further explored,
			False otherwise, i.e, there exists arguments to prune that path.
		"""
		flag = True
		if temp_reward < self.total_reward:
			flag = False
			self.bound_counter.count()
		return flag

	def roll_out_search(self, rollout_path, rollout_reward, depth):
		"""
		Performs the rollout search.

		If the depth is greater or equal than the max depth or the current state is terminal
		(i.e., is a coloring) the function tries to update the current incumbent temporary path, i.e., calls
		update_temporary_search. Otherwise, it recursively explores all the next possible paths by calling the
		roll_out_search for all the following states.

		Note that the recursion eventually ends because the depth argument is increased by 1 every time the
		function is called.

		Parameters
		----------
		rollout_path:
			The current temporary path. A sequence of state and action paris that indicates the path used.
		rollout_reward:
			Temporary number of colors used.
		depth:
			The current exploration depth.
		"""
		if depth >= self.max_depth or rollout_path[-1].is_coloring(soft=True):
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
		"""
		Initializes the local search.

		This is done by calling the roll_out_search method at _temp_path [state], with
		a _temp_reward of -inf and a depth of 0.

		Parameters
		----------
		state:
			State at which the local search will be initialized.

		Returns
		-------
			The action to be taken after performing the local rollout search,
			i.e., the action that maximizes the self.max_depth rollout minimization problem.
		"""
		self._temp_path = [state]
		self._temp_reward = float('-inf')

		self.roll_out_search(self._temp_path, len(state), 0)

		return self._temp_path[1]

	def solve(self):
		"""
		Finds a Coloring for the given graph using the rollout policy.

		In every step it calls the roll_out function to build the rollout policy.
		Returns
		-------
		Coloring:
			The coloring that results of running the rollout policy with the given heuristic.
		"""
		coloring = self.env.reset()
		done = False
		
		while not done:
			action = self.roll_out(coloring)
			coloring, cost, done, info = self.env.step(action)

		return coloring







