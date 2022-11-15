import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations, product
from scipy.special import softmax

from causal_env_v0 import *
from gpt3 import gpt3_scoring
from utils import _enum, _verbalize_action


class BaseAgent(object):
    def __init__(self, n_objects) -> None:
        self.n_objects = n_objects

    def next_action(self, state_info):
        raise NotImplementedError()


class RandomAgent(BaseAgent):
    def __init__(self, n_objects, seed=42) -> None:
        super().__init__(n_objects)
        self.random = np.random.RandomState(seed)

    def next_action(self, state_info):
        return self.random.randint(0, 2, size=self.n_objects)


class GPT3Agent(BaseAgent):
    def __init__(
        self,
        n_objects,
        objects,
        random_actions=False,
        argmax_action=True,
        engine="text-davinci-002",
        seed=42,
        verbose=False,
        output_dir=".",
    ) -> None:
        super().__init__(n_objects)

        # Agent configuration
        self.random = np.random.RandomState(seed)
        self.engine = engine

        # Prompt configuration
        self.objects = objects
        self.action_prompt = "Experiment {exp_id}: We place the following objects on the detector and observe the outcome.\nAction: "
        self.prompt = (
            open("prompt_header.txt", "r")
            .read()
            .format(objects=", ".join(self.objects))
        )

        # Action configuration
        self.actions = [
            np.array(a) for a in product(*[[0, 1]] * self.n_objects)
        ]  # possible actions
        self.random.shuffle(self.actions)  # shuffle to make sure no bias
        self.actions_text = [_verbalize_action(a, self.objects) for a in self.actions]
        self.random_actions = random_actions
        self.argmax_action = argmax_action
        self.output_dir = output_dir
        self.verbose = verbose

        # Agent state variables
        self.stop_proba = []
        self.exp_id = 2

    def _is_decidable(self):
        return softmax(
            gpt3_scoring(
                self.prompt
                + f"Based on your experiments, are you able to identify all the blickets? ",
                options=["Yes", "No"],
                lock_token="?",
                engine=self.engine,
            )
        )[0]

    def _score_actions(self):
        return softmax(
            gpt3_scoring(
                self.prompt + self.action_prompt.format(exp_id=self.exp_id),
                self.actions_text,
                engine=self.engine,
                lock_token=":",
            )
        )

    def _score_blickets(self):
        """
        Score possible sets of blickets using the model.
        If the model doesn't support scoring, just use completion and use score=0 for other solutions.

        """
        blicket_choices = [
            _enum([f"the {o}" for o in combo], final="and") + " are blickets."
            for combo in [
                list(c)
                for i in range(2, len(self.objects) + 1)
                for c in combinations(self.objects, i)
            ]
        ] + [f"the {o} is a blicket." for o in self.objects]
        blicket_scores = softmax(
            gpt3_scoring(
                self.prompt
                + f"Based on your experiments, which objects are blickets? ",
                options=blicket_choices,
                lock_token="?",
                engine=self.engine,
            )
        )
        return blicket_choices, blicket_scores

    def _score_blickets_individually(self):
        object_scores = np.zeros(len(self.objects))
        for i, o in enumerate(self.objects):
            object_scores[i] = softmax(
                gpt3_scoring(
                    self.prompt
                    + f"Based your experiments, is the {o} part of the set of blickets? ",
                    options=[f"Yes", "No"],
                    lock_token="?",
                    engine=self.engine,
                )
            )[0]
        return object_scores

    def decide_blickets(self):
        blicket_scores = self._score_blickets_individually()
        return blicket_scores > 0.5

    def next_action(self, state_info):

        # Add previous action and outcome to the prompt
        self.prompt += (
            self.action_prompt.format(exp_id=self.exp_id - 1)
            + f"{_verbalize_action(state_info['prev_action'], self.objects)}\n"
            + f"Outcome: The detector {'did' if state_info['detector_activated'] else 'did not'} turn on.\n\n"
        )
        self.verbose and print(self.prompt)

        # Query the model about its beliefs about blickets
        object_combos, object_combo_scores = self._score_blickets()
        object_indiv_scores = self._score_blickets_individually()

        # Model belief that it has enough info to decide all blickets
        self.stop_proba.append(self._is_decidable())

        # Query model for next action
        if self.stop_proba[-1] > 0.98:
            # -- Model wants to stop exploring
            best_action = None  # Stop action
        else:
            # -- Model wants to keep exploring
            if self.random_actions:
                action_scores = self.random.rand(len(self.actions))
            else:
                action_scores = self._score_actions()

            # Use the highest-scored action or sample from the score distribution
            if self.argmax_action:
                best_action = self.actions[np.argmax(action_scores)]
            else:
                best_action = self.actions[
                    self.random.choice(np.arange(len(self.actions)), p=action_scores)
                ]

        # Plot multiple figures
        # -- Model belief that it has all required info to decide blickets
        plt.clf()
        plt.plot(np.arange(1, self.exp_id), self.stop_proba)
        plt.xlabel("Step")
        plt.ylabel("P(enough info to answer)")
        plt.savefig(f"{self.output_dir}/stop_proba.png", bbox_inches="tight")
        # -- Blick scores (individually per object)
        plt.clf()
        sns.barplot(x=np.arange(len(object_indiv_scores)), y=object_indiv_scores)
        plt.gca().set_xticklabels(self.objects)
        plt.axhline(0.5, linestyle="--", color="red")
        plt.ylim(0, 1)
        plt.ylabel("P(blicket)")
        plt.xlabel("Object")
        plt.title("Model belief that objects are blickets")
        plt.savefig(f"{self.output_dir}/object_scores_{self.exp_id - 1}.png")
        # -- Blicket scores (for combinations of objects)
        plt.clf()
        sns.barplot(x=np.arange(len(object_combos)), y=object_combo_scores)
        plt.gca().set_xticklabels(object_combos, rotation=90)
        plt.ylabel("P(blicket)")
        plt.xlabel("Blicket set")
        plt.title("Model belief that objects are blickets")
        plt.savefig(
            f"{self.output_dir}/object_scores_combined_{self.exp_id - 1}.png",
            bbox_inches="tight",
        )
        if best_action is not None:
            # -- Action scores
            plt.clf()
            sns.barplot(x=np.arange(len(action_scores)), y=action_scores)
            plt.xlabel("Action (on/off for each object)")
            plt.ylabel("P(next best action)")
            plt.title("Model belief of the next best action to perform")
            plt.gca().set_xticklabels([str(a) for a in self.actions])
            plt.savefig(f"{self.output_dir}/action{self.exp_id}_scores.png")

        self.exp_id += 1
        return best_action
