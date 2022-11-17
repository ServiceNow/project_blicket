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

    def decide_blickets(self):
        raise NotImplementedError()


class LLMScoringAgent(BaseAgent):
    """
    A base class for scoring-based language model agents.

    Notes: One only needs to define the self._scoring function to get a functional agent.

    """

    def __init__(
        self,
        n_objects,
        objects,
        random_actions=False,
        argmax_action=True,
        stop_proba_threshold=0.97,
        seed=42,
        verbose=False,
        minimize_prompting=False,
        output_dir=".",
    ) -> None:
        super().__init__(n_objects)

        # Agent configuration
        self.random = np.random.RandomState(seed)

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
        self.stop_proba_threshold = stop_proba_threshold
        self.output_dir = output_dir
        self.verbose = verbose
        self.minimize_prompting = minimize_prompting

        # Agent state variables
        self.stop_proba = []
        self.exp_id = 2
        self.last_action = None
        self.requested_stop = False

    def _score_options(self, prompt, options, lock_token=None):
        """
        Score multiple options using language model. Needs to be implemented for each model individually.

        """
        raise NotImplementedError()

    def _is_decidable(self):
        """
        Get the model's confidence that the blickets are identifiable using the information gathered until now

        """
        return self._score_options(
            prompt=self.prompt
            + f"Based on your experiments, are you able to identify all the blickets? ",
            options=["Yes", "No"],
            lock_token="?",
        )[0]

    def _plot_blicket_status(self, final=False):
        """
        Plot the model's belief that each object is a blicket

        """
        # -- Blick scores (individually per object)
        object_indiv_scores = self._score_blickets_individually()
        plt.clf()
        sns.barplot(x=np.arange(len(object_indiv_scores)), y=object_indiv_scores)
        plt.gca().set_xticklabels(self.objects)
        plt.axhline(0.5, linestyle="--", color="red")
        plt.ylim(0, 1)
        plt.ylabel("P(blicket)")
        plt.xlabel("Object")
        plt.title("Model belief that objects are blickets")
        plt.savefig(
            f"{self.output_dir}/object_scores_{self.exp_id - 1 if not final else 'final'}.png"
        )
        # -- Blicket scores (for combinations of objects)
        object_combos, object_combo_scores = self._score_blickets()
        plt.clf()
        sns.barplot(x=np.arange(len(object_combos)), y=object_combo_scores)
        plt.gca().set_xticklabels(object_combos, rotation=90)
        plt.ylabel("P(blicket)")
        plt.xlabel("Blicket set")
        plt.title("Model belief that objects are blickets")
        plt.savefig(
            f"{self.output_dir}/object_scores_combined_{self.exp_id - 1 if not final else 'final'}.png",
            bbox_inches="tight",
        )

    def _plot_stop_proba(self):
        plt.clf()
        plt.plot(np.arange(1, self.exp_id), self.stop_proba)
        plt.xlabel("Step")
        plt.ylabel("P(enough info to answer)")
        plt.savefig(f"{self.output_dir}/stop_proba.png", bbox_inches="tight")

    def _score_actions(self):
        """
        Get model belief on the next best action

        """
        return self._score_options(
            prompt=self.prompt + self.action_prompt.format(exp_id=self.exp_id),
            options=self.actions_text,
            lock_token=":",
        )

    def _score_blickets(self):
        """
        Get model confidence that every possible set of objects is a blicket

        """
        blicket_choices = [
            _enum([f"the {o}" for o in combo], final="and") + " are blickets."
            for combo in [
                list(c)
                for i in range(2, len(self.objects) + 1)
                for c in combinations(self.objects, i)
            ]
        ] + [f"the {o} is a blicket." for o in self.objects]
        blicket_scores = self._score_options(
            self.prompt + f"Based on your experiments, which objects are blickets? ",
            options=blicket_choices,
            lock_token="?",
        )
        return blicket_choices, blicket_scores

    def _score_blickets_individually(self):
        """
        Get model confidence that each object is a blicket (individually)

        """
        object_scores = np.zeros(len(self.objects))
        for i, o in enumerate(self.objects):
            object_scores[i] = self._score_options(
                prompt=self.prompt
                + f"Based your experiments, is the {o} part of the set of blickets? ",
                options=[f"Yes", "No"],
                lock_token="?",
            )[0]
        return object_scores

    def _update_prompt(self, prev_action, detector_activated):
        """
        Update prompt with feedback from previous experiment

        """
        self.prompt += (
            self.action_prompt.format(exp_id=self.exp_id - 1)
            + f"{_verbalize_action(prev_action, self.objects)}\n"
            + f"Outcome: The detector {'did' if detector_activated else 'did not'} turn on.\n\n"
        )

    def decide_blickets(self):
        blicket_scores = self._score_blickets_individually()
        return blicket_scores > 0.5

    def close(self, state_info):
        self._update_prompt(**state_info)
        self._plot_blicket_status(final=True)
        self.stop_proba.append(self._is_decidable())
        self._plot_stop_proba()

    def next_action(self, state_info):
        """
        Gives feedback on previous action to the agent and asks for next action

        Parameters:
        -----------
        state_info: dict
            A dictionnary with keys: 1) prev_action: the previous action in array format,
            2) detector_activated: whether or not the dector was activated by previous action.

        Returns:
        --------
        action: np.ndarray, dtype=uint, shape=(self.n_objects,)
            An array of binary values, where 1 means that the object is placed on the detector
            and 0 means that it is not.

        """
        # Add previous action and outcome to the prompt
        self._update_prompt(**state_info)
        self.verbose and print(self.prompt)

        # Model belief that it has enough info to decide all blickets
        self.stop_proba.append(self._is_decidable())

        # Query model for next action
        if self.stop_proba[-1] > self.stop_proba_threshold or self.requested_stop:
            # -- Model wants to stop exploring
            best_action = None  # Stop action
            self.requested_stop = True  # Ensure no next actions will be made
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
        self._plot_stop_proba()
        # -- Blicket scores
        if not self.minimize_prompting:
            self._plot_blicket_status()
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
        self.last_action = best_action
        return best_action


class GPT3Agent(LLMScoringAgent):
    def __init__(
        self,
        n_objects,
        objects,
        random_actions=False,
        argmax_action=True,
        stop_proba_threshold=0.97,
        engine="text-davinci-002",
        seed=42,
        verbose=False,
        minimize_prompting=False,
        output_dir=".",
    ) -> None:
        super().__init__(
            n_objects,
            objects,
            random_actions,
            argmax_action,
            stop_proba_threshold,
            seed,
            verbose,
            minimize_prompting,
            output_dir,
        )

        # Agent configuration
        self.engine = engine

    def _score_options(self, prompt, options, lock_token=None):
        return softmax(
            gpt3_scoring(
                prompt,
                options,
                engine=self.engine,
                lock_token=lock_token,
            )
        )
