"""
Taken from https://github.com/CannyLab/causal_overhypotheses/
MIT license (https://github.com/CannyLab/causal_overhypotheses/blob/master/LICENSE)

"""
import gym
import numpy as np
import itertools
import random

from gym import spaces

from typing import Dict, Any, Tuple, List, Set


class Hypothesis:
    @property
    def blickets(self) -> Set[int]:
        ...

    @classmethod
    def test(cls, blickets: Set[int]) -> bool:
        ...


"""
Blabla

asdas


"""


class ConjunctiveHypothesis:
    blickets = None
    structure = "conjunctive"

    @classmethod
    def test(cls, blickets: Set[int]) -> bool:
        return all(c in blickets for c in cls.blickets)  # type: ignore


class ABconj(ConjunctiveHypothesis):
    blickets = set([0, 1])


class ACconj(ConjunctiveHypothesis):
    blickets = set([0, 2])


class BCconj(ConjunctiveHypothesis):
    blickets = set([1, 2])


class ABCconj(ConjunctiveHypothesis):
    blickets = set([0, 1, 2])


### Base Disjunctive Hypotheses for 3 blickets ###


class DisjunctiveHypothesis:
    blickets = None
    structure = "disjunctive"

    @classmethod
    def test(cls, blickets: Set[int]) -> bool:
        return any(c in blickets for c in cls.blickets)  # type: ignore


class Adisj(DisjunctiveHypothesis):
    blickets = set([0])


class Bdisj(DisjunctiveHypothesis):
    blickets = set([1])


class Cdisj(DisjunctiveHypothesis):
    blickets = set([2])


class ABdisj(DisjunctiveHypothesis):
    blickets = set([0, 1])


class ACdisj(DisjunctiveHypothesis):
    blickets = set([0, 2])


class BCdisj(DisjunctiveHypothesis):
    blickets = set([1, 2])


class ABCdisj(DisjunctiveHypothesis):
    blickets = set([0, 1, 2])


class CausalEnv_v0(gym.Env):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        """
        Representation of the Blicket environment, based on the exeperiments presente in the causal learning paper.

        Args:
            env_config (Dict[str, Any]): A dictionary representing the environment configuration.
                Keys: Values (Default)


        Action Space:
            => [Object A (on/off), Object B state (on/off), Object C state (on/off)]

        """

        # ---------------------------------------
        #       Environment configuration
        # ---------------------------------------

        # The total number of objects, but not necessarily blickets (seems like a mistake from initial authors)
        # The actual number of blickets is determined by the selected hypothesis.
        self._n_blickets = env_config.get("n_blickets", 3)

        # When symbolic mode is activated, the blicket colors are returned instead of simply their status (on/off)
        # This is what they used for the RL experiments.
        self._symbolic = True  # XXX: Keeping this for future reference, but I deleted all the non-symbolic code below
        # The following seems to be the dimensions used to sample colors for the blickets. I don't really understand
        # but it's used only in non-symbolic mode, which we don't use.
        self._blicket_dim = -1  # Disabled but kept for future reference.

        self._reward_structure = env_config.get(
            "reward_structure", "baseline"
        )  # Start with baseline reward structure

        # Different kind of tasks
        if self._reward_structure not in (
            "baseline",  # Full-on exploration, positive reward for lighting up the detector, which reduces with n_steps
            "quiz",  # Exploration phase followed by an evaluation phase where need to guess which obj are blickets
            "quiz-type",  # Same as above, but also need to guess the type of overhypothesis
            "quiz-typeonly",  # Exploration phase followed by evaluation phase where need to guess overhypothesis type
        ):
            raise ValueError(
                "Invalid reward structure: {}, must be one of (baseline, quiz, quiz-type, quiz-typeonly)".format(
                    self._reward_structure
                )
            )

        # Setup penalties and reward structures
        # Whether or not to use the step reward penalty (below)
        self._add_step_reward_penalty = env_config.get("add_step_reward_penalty", False)
        # Looks like a penalty that reduces the reward obtained at each step to favor shorter trajectories
        self._step_reward_penalty = env_config.get("step_reward_penalty", 0.01)
        # Whether or not to increment the reward based on the state of the detector when model switches to eval mode
        self._add_detector_state_reward_for_quiz = env_config.get(
            "add_detector_state_reward_for_quiz", False
        )
        # The amount of reward received when the detector turns on
        self._detector_reward = env_config.get("detector_reward", 1)
        # Reward received for correctly guessing blicket status of an object in the evaluation phase of the quiz
        self._quiz_positive_reward = env_config.get("quiz_positive_reward", 1)
        # Reward received for incorrectly guessing blicket status of an object in the evaluation phase of the quiz
        self._quiz_negative_reward = env_config.get("quiz_negative_reward", -1)
        # Maximum number of exploration steps (for quiz tasks, eval mode is auto triggerred after max_steps)
        self._max_steps = env_config.get("max_steps", 20)
        # How long to stay in exploration mode even if the model activated exploration mode (forces it)
        self._quiz_disabled_steps = env_config.get("quiz_disabled_steps", -1)

        # ---------------------------------------
        #          Gym environment setup
        # ---------------------------------------

        # XXX: In quiz mode, there is an extra action dimension corresponding to activating the evaluation phase
        self.action_space = (
            spaces.MultiDiscrete([2] * self._n_blickets)
            if "quiz" not in self._reward_structure
            else spaces.MultiDiscrete([2] * (self._n_blickets + 1))
        )

        # XXX: In quiz mode, there is an extra state dimension denoting if we are in evaluation mode or not
        #      I think that this dimension is used since the env doesn't necessarily transition to eval mode
        #      when the model requests it. There is code to force exploring for some amount of steps.
        if "quiz" in self._reward_structure:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(self._n_blickets + 2,), dtype=np.float32
            )  # The state of all of the blickets, plus the state of the detector plus the quiz indicator
        else:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(self._n_blickets + 1,), dtype=np.float32
            )  # The state of all of the blickets, plus the state of the detector

        # ---------------------------------------
        # Configure underlying SCM
        # ---------------------------------------

        # Defines the potential structure of the ground truth SCM (they sample one randomly)
        self._hypotheses: List[Hypothesis] = env_config.get(
            "hypotheses",
            [
                ABconj,
                ACconj,
                BCconj,
                # ABCconj,
                Adisj,
                Bdisj,
                Cdisj,
                # ABdisj,
                # ACdisj,
                # BCdisj,
                # ABCdisj,
            ],
        )

        # Randomly select one hypothesis from the list of possible ones
        self._current_gt_hypothesis = random.choice(self._hypotheses)

        # ---------------------------------------
        #  Trackers for the environment's state
        # ---------------------------------------
        self._steps = 0
        self._observations = 0
        self._quiz_step = None
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the environment

        """
        # Randomly select a new hypothesis (i.e., a ground truth SCM)
        self._current_gt_hypothesis = random.choice(self._hypotheses)

        # Reset the step trackers
        self._steps = 0
        self._quiz_step = None

        # Get the baseline observation (i.e., there are no blickets on the detector)
        return self._get_observation(blickets=np.zeros(self._n_blickets))

    def _get_baseline_observation(self, blickets: np.ndarray) -> np.ndarray:
        """
        Get observation for the baseline task, which is just exploration to turn on the detector without a quiz

        Parameters:
        -----------
        blickets: np.ndarray, shape=(n_blickets,)
            A binary array containing the state of each blicket (on [1] or off [0] of the detector)

        Returns:
        --------
        observation: np.ndarray, shape=TODO:
            TODO:

        """
        # Produces an observation composed of the state of each blicket + a binary indicating if the detector is on or not.
        return np.concatenate(
            [
                blickets,
                np.array([1]) if self._get_detector_state(blickets) else np.array([0]),
            ],
            axis=0,
        )  # type: ignore

    def _get_quiz_observation(self, blickets: np.ndarray) -> np.ndarray:
        """
        Get an observation in quiz mode

        Parameters:
        -----------
        blickets: np.ndarray, shape=(n_blickets,)
            A binary array containing the state of each blicket (on [1] or off [0] of the detector)

        Returns:
        --------
        observation: np.ndarray, shape=TODO:
            TODO:

        """
        if self._quiz_step is not None:
            # Quiz phase
            return np.concatenate(
                [
                    np.array(
                        [
                            1 if self._quiz_step == i else 0
                            for i in range(self._n_blickets)
                        ]
                    ),
                    np.array(
                        [0]
                    ),  # Detector state: 0 since it's irrelevant here (asking if objects are blickets)
                    np.array([1]),  # Quiz indicator (1 since we are in quiz phase)
                ],
                axis=0,
            )

        # Exploration phase: the quiz phase needs to be activated via an action or after some number of steps has passed
        return np.concatenate(
            [
                blickets,  # Blickets
                np.array(
                    [1] if self._get_detector_state(blickets) else [0]
                ),  # Detector state
                np.array([0]),  # Quiz indicator (we are not in quiz phase, so 0)
            ],
            axis=0,
        )  # type: ignore

    def _get_observation(self, blickets: np.ndarray) -> np.ndarray:
        """
        Get appropriate observation type based on the task (specified via _reward structure)

        """
        if self._reward_structure == "baseline":
            return self._get_baseline_observation(blickets)
        elif "quiz" in self._reward_structure:
            return self._get_quiz_observation(blickets)
        raise ValueError("Invalid reward structure: {}".format(self._reward_structure))

    def _get_detector_state(self, active_blickets: np.ndarray) -> bool:
        """
        Get's the state of the detector's light based on the blickets that are currently placed
        on the machine.

        Parameters:
        -----------
        active_blickets: np.ndarray, shape=(n_blickets,)
            A binary array indicating if any of the blickets is on (1) or off (0) the detector.

        Returns:
        --------
        detector_state: int
            0 if detector light is off and 1 if detector light is on.

        """
        blickets_on = set()
        for i in range(len(active_blickets)):
            if active_blickets[i] == 1:
                blickets_on.add(i)
        return self._current_gt_hypothesis.test(blickets_on)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Make an action

        """

        observation, reward, done, info = (None, 0, False, {})

        # Generate the observations and reward
        if self._reward_structure == "baseline":
            observation = self._get_baseline_observation(action[: self._n_blickets])

            # Get the reward
            if self._add_step_reward_penalty:
                reward -= self._step_reward_penalty
            if self._get_detector_state(action[: self._n_blickets]):
                reward += self._detector_reward

            # TODO: The step is incremented at the end of this function. Why isn't that check done before calculating rewards
            #       it seems like this would allow to do one extra step at the end of the execution.
            done = self._steps > self._max_steps

        elif "quiz" in self._reward_structure:
            if self._quiz_step is not None:
                # Evaluation mode: model needs to claim which objects are blickets

                # Quiz where the model only need to guess which objects are blickets
                if self._reward_structure == "quiz":
                    # The quiz step indicates which object we are evaluating at the current time
                    # The model must put action[_quiz_step] = 1 for the _quiz_step'th object if it
                    # believes that it is a blicket and 0 otherwise.
                    # Reward: _quiz_positive_reward if correct, _quiz_negative_reward if incorrect.
                    reward = (
                        self._quiz_positive_reward
                        if (
                            action[self._quiz_step] == 1
                            and self._quiz_step in self._current_gt_hypothesis.blickets
                            or action[self._quiz_step] == 0
                            and self._quiz_step
                            not in self._current_gt_hypothesis.blickets
                        )
                        else self._quiz_negative_reward
                    )

                # Quiz where the model also needs to guess the type of overhypothesis
                elif self._reward_structure in ("quiz-type", "quiz-typeonly"):
                    # Start by asking the model to identify all the blickets (one by one)
                    if self._quiz_step < self._n_blickets:
                        reward = (
                            self._quiz_positive_reward
                            if (
                                action[self._quiz_step] == 1
                                and self._quiz_step
                                in self._current_gt_hypothesis.blickets
                                or action[self._quiz_step] == 0
                                and self._quiz_step
                                not in self._current_gt_hypothesis.blickets
                            )
                            else self._quiz_negative_reward
                        )
                    else:
                        # Ask the model to guess the overhypothesis type by setting the first element of
                        # the action vector to zero or one. This is a bit strange since in previous iterations,
                        # this action was used to identify if the first object was a blicket. Semantics change.
                        reward = (
                            0.5  # Review: the reward and negative rewards are hardcoded for this task
                            if (
                                action[0] == 0
                                and issubclass(
                                    self._current_gt_hypothesis, ConjunctiveHypothesis
                                )
                                or action[0] == 1
                                and issubclass(
                                    self._current_gt_hypothesis, DisjunctiveHypothesis
                                )
                            )
                            else -0.5
                        )

                # Increment the quiz step, i.e., move to following object
                self._quiz_step += 1
                observation = self._get_quiz_observation(action[: self._n_blickets])

                if self._reward_structure in ("quiz-type", "quiz-typeonly"):
                    if self._quiz_step > self._n_blickets:
                        done = True
                elif self._reward_structure == "quiz":
                    if self._quiz_step >= self._n_blickets:
                        done = True

            else:
                # Check the action to see if we should go to quiz phase
                if self._steps > self._max_steps or (
                    action[-1] == 1 and self._steps > self._quiz_disabled_steps
                ):
                    # Transition: we go to evaluation phase
                    if self._add_step_reward_penalty:
                        reward -= self._step_reward_penalty
                    if (
                        self._add_detector_state_reward_for_quiz
                        and self._get_detector_state(action[: self._n_blickets])
                    ):
                        reward += self._detector_reward

                    # Activate evaluation mode
                    self._quiz_step = (
                        0  # Start by asking about the first object (blicket or not)
                        if self._reward_structure != "quiz-typeonly"
                        else self._n_blickets  # This makes sure the code will start at type-checking phase
                    )
                    observation = self._get_quiz_observation(action[: self._n_blickets])
                else:
                    # Transition: we stay in exploration phase
                    observation = self._get_quiz_observation(action[: self._n_blickets])

        assert observation is not None

        self._steps += 1
        return observation, reward, done, info
