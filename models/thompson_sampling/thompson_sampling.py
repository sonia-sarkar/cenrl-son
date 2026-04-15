import collections
import typing
from statistics import geometric_mean

import networkx as nx
import numpy as np

from models.base.action_space import ActionSpaceBase, NODE_TYPE_KEY, POSSIBLE_TARGET_FEATURES, SLEEPING, ACTION_ATTEMPTS
from models.base.model import Model, ParserOptions, run_multiprocessing


class ThompsonSamplingActionSpace (ActionSpaceBase):
    SUCCESS_VALUE = "success_value"
    FAILURE_VALUE = "failure_value"

    def build_graph(self) -> nx.DiGraph:
        g = super().build_graph()
        # initialize to 1 because thompson sampling uses successes / failures + 1 each.
        for n, n_data in g.nodes(data=True):
            n_data[ThompsonSamplingActionSpace.SUCCESS_VALUE] = 1
            n_data[ThompsonSamplingActionSpace.FAILURE_VALUE] = 1

        return g


class ThompsonSampling(Model):

    def is_optimal_action(self, selected_arm_seq: typing.List[str]) -> bool:
        # no way to figure this out for now
        return False

    def update_optimal_value(self):
        # no need to
        pass

    def choose_arm(self) -> typing.List[str]:
        def _choose_arm_by_layer(_node_keys: typing.List[str],
                                 _success_values: np.array,
                                 _failure_values: np.array) -> str:
            samples = np.random.beta(_success_values, _failure_values)
            selected_index = np.nanargmax(samples)
            return _node_keys[selected_index]

        # loop through the hierarchy and choose largest Q (stop when we reach target nodes)
        source = self.action_space.get_root()
        selected_arm = None
        reached_target_nodes = False
        selected_arms_history = []

        while not reached_target_nodes:
            immediate_children = []
            success_values = []
            failure_values = []

            for succ in self.action_space.get_graph().successors(source):
                succ_data = self.action_space.get(succ)
                if succ_data[NODE_TYPE_KEY] in POSSIBLE_TARGET_FEATURES:
                    reached_target_nodes = True
                    break

                immediate_children.append(succ)
                success_values.append(succ_data[ThompsonSamplingActionSpace.SUCCESS_VALUE])
                failure_values.append(succ_data[ThompsonSamplingActionSpace.FAILURE_VALUE])

            if reached_target_nodes:
                break

            if immediate_children and not reached_target_nodes:
                selected_arm = _choose_arm_by_layer(immediate_children,
                                                    np.array(success_values),
                                                    np.array(failure_values))
                selected_arms_history.append(selected_arm)

            source = selected_arm
            # print(f"{self.current_epoch_num}: chose arm {selected_arm}")

        # NOTE: must update the ACTION_ATTEMPTS for all arms except for last (which will be updated in the observe)
        for a in selected_arms_history[:-1]:
            self.action_space.get(a)[ACTION_ATTEMPTS] += 1

        self.last_selected_arm_index = selected_arms_history[-1]
        return selected_arms_history

    def propagate_rewards(self, selected_arm: str):
        d = collections.deque()
        d.append(selected_arm)

        while d:
            curr_node = d.popleft()

            # ignore root node
            if curr_node == self.action_space.get_root():
                continue

            # if it does not only have target successors, then calculate its aggregated q_value
            if not self.action_space.has_target_successors(curr_node):
                successors = self.action_space.get_active_nontarget_successors(curr_node)
                if successors:
                    success_values = [self.action_space.get(x)[ThompsonSamplingActionSpace.SUCCESS_VALUE] for x in successors]
                    agg_success_value = geometric_mean(success_values)
                    self.action_space.get(curr_node)[ThompsonSamplingActionSpace.SUCCESS_VALUE] = agg_success_value
                    #print(f"agg success value {agg_success_value} for {curr_node}")

                    failure_values = [self.action_space.get(x)[ThompsonSamplingActionSpace.FAILURE_VALUE] for x in successors]
                    agg_failure_value = geometric_mean(failure_values)
                    self.action_space.get(curr_node)[ThompsonSamplingActionSpace.FAILURE_VALUE] = agg_failure_value
                    #print(f"agg failure value {agg_failure_value} for {curr_node}")

            # add parents
            for p in self.action_space.get_graph().predecessors(curr_node):
                d.append(p)

    def observe(self, selected_arm: str, measurement_result: float) -> float:
        self.action_space.get(selected_arm)[ACTION_ATTEMPTS] += 1
        self.action_space.get(selected_arm)[ThompsonSamplingActionSpace.SUCCESS_VALUE] += measurement_result
        self.action_space.get(selected_arm)[ThompsonSamplingActionSpace.FAILURE_VALUE] += (1 - measurement_result)
        return measurement_result


class ThompsonSamplingParserOptions(ParserOptions):
    def add_arguments(self):
        super().add_arguments()

    def set_params(self, args):
        super().set_params(args)
        self.params["action_value_file"] = None


if __name__ == "__main__":
    parser = ThompsonSamplingParserOptions()
    params = parser.parse()

    addition_model_run_kwargs = {"action_space_klass": ThompsonSamplingActionSpace}

    run_multiprocessing(ThompsonSampling, params,
                        addition_model_run_kwargs=addition_model_run_kwargs)
