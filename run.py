import argparse
import numpy as np
import os
import pandas as pd

from collections import defaultdict
from enum import Enum
from scipy.special import softmax

from agents import GPT3Agent
from causal_env_v0 import *
from utils import _score_blickets


class BlicketStatus(Enum):
    unknown = 0
    maybe = 1
    yes = 2
    no = 3


def generate_objects(n_objects, random_state=np.random.RandomState()):
    shapes = [
        "cone",
        "cube",
        "cylinder",
        "dome",
        "frustum",
        "prism",
        "pyramid",
        "sphere",
        "torus",
    ]

    colors = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "teal",
        "magenta",
        "brown",
        "yellow",
        "pink",
        "black",
        "white",
    ]

    # The following ensures unique color and shape for each object.
    return [
        f"{c} {s}"
        for c, s in zip(
            random_state.choice(colors, size=n_objects, replace=False),
            random_state.choice(shapes, size=n_objects, replace=False),
        )
    ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Runtime")
    parser.add_argument("--n-objects", type=int, default=3)
    parser.add_argument(
        "--random-actions", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--stop-on-answerable", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--minimize-prompting", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--experiment-dir", type=str, default=".")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--gpt3-engine", type=str, default="text-davinci-002")

    args = parser.parse_args()
    print("Configuration:", args)
    n_objects = args.n_objects
    random_actions = args.random_actions
    gpt3_engine = args.gpt3_engine
    max_steps = args.max_steps
    repeats = args.repeats
    minimize_prompting = args.minimize_prompting
    experiment_dir = args.experiment_dir
    stop_on_answerable = args.stop_on_answerable
    random = np.random.RandomState(args.random_seed)

    env = CausalEnv_v0(
        {
            "reward_structure": "baseline",
            "hypotheses": [
                ABconj
            ],  # We fully control the hypothesis (no sampling among a list)
            "n_blickets": n_objects,
        }
    )

    # Repeats keep the environment fixed but change the object names
    results = []
    for repeat in range(repeats):
        repeat_dir = f"{experiment_dir}/repeat_{repeat}"
        os.makedirs(repeat_dir, exist_ok=True)

        objects = generate_objects(n_objects, random_state=random)
        print("The objects are:", ", ".join(objects))
        print()
        print(
            "Ground truth: the blickets are the",
            ", ".join([objects[i] for i in env._current_gt_hypothesis.blickets]),
            env._current_gt_hypothesis,
            f"and the detector is {'conjunctive' if issubclass(env._current_gt_hypothesis, ConjunctiveHypothesis) else 'disjunctive'}.",
        )
        print()

        # Memory to keep track of what we know
        blicket_status = defaultdict(lambda: BlicketStatus.unknown)
        is_answerable = False  # Track if we have gathered enough info to make a call on each blicket (in theory)

        # Define the initial state of the system (shown as an experiment to the agent)
        action = np.ones(n_objects)
        detector_activated = True

        # Instantiate the agent
        agent = GPT3Agent(
            n_objects=n_objects,
            objects=objects,
            random_actions=random_actions,
            stop_proba_threshold=0.97
            if not stop_on_answerable
            else np.inf,  # Never stop on model request
            engine=gpt3_engine,
            output_dir=repeat_dir,
            seed=repeat,
            verbose=False,
            minimize_prompting=minimize_prompting,
        )
        action_history = [action]

        for j in range(max_steps):
            if all(
                blicket_status[idx] in [BlicketStatus.yes, BlicketStatus.no]
                for idx in range(n_objects)
            ):
                is_answerable = True
                if stop_on_answerable:
                    break

            action = agent.next_action(
                state_info={
                    "prev_action": action,
                    "detector_activated": detector_activated,
                }
            )
            action_history.append(action)
            if action is None:
                print("Model requested stop.")
                break

            n_obs, reward, done, info = env.step(action)
            detector_activated = n_obs[-1] == 1

            object_on_detector = action[:n_objects] == 1

            print(f"Step: {j + 1} -- Gathered enough info to answer? {is_answerable}")
            print(
                "|_________ Action: place",
                [objects[i] for i, a in enumerate(action[:n_objects]) if a == 1],
                "on the detector.",
            )
            if n_obs[-1] == 1:
                print(
                    "|_________ Reward: The detector did turn on (DETECTOR ACTIVATED)."
                )
            else:
                print("|_________ Reward: The detector did not turn on.")

            # Update our knowledge of blick statuses
            if issubclass(env._current_gt_hypothesis, ConjunctiveHypothesis):
                if detector_activated:
                    for idx, on_detector in enumerate(object_on_detector):
                        # If object is off and detector activated, it's not a blicket
                        if not on_detector:
                            blicket_status[idx] = BlicketStatus.no

                        # If object is on and detector is activated, it might be a blicket
                        elif blicket_status[idx] == BlicketStatus.unknown:
                            blicket_status[idx] = BlicketStatus.maybe

                else:
                    # We can tell that an object is a blicket for sure if:
                    # * it was previously seen on an activated detector (so status is maybe)
                    # * the detector is not activated and the object is not on the detector
                    # * all other objects that are off are known not to be blickets (otherwise
                    #   the fact that the detector is off could be attributed to another object,
                    #   which is a blicket, not being on the detector.)
                    for idx, on_detector in enumerate(object_on_detector):
                        if (
                            blicket_status[idx] == BlicketStatus.maybe
                            and not on_detector
                            and all(
                                blicket_status[idx_] == BlicketStatus.no
                                for idx_ in range(n_objects)
                                if idx_ != idx and not object_on_detector[idx_]
                            )
                        ):
                            blicket_status[idx] = BlicketStatus.yes

            else:
                raise NotImplementedError("Disjunction not supported yet.")

            print(
                "|_________ Blicket status:",
                [blicket_status[idx] for idx in range(n_objects)],
            )

        print("Exploration complete.")
        print(action_history)
        agent_blickets = agent.decide_blickets()
        print(agent_blickets)
        metrics = _score_blickets(agent_blickets, env._current_gt_hypothesis.blickets)
        print(metrics)
        print("\n" * 2)

        result = dict(vars(args))
        result.update(
            dict(repeat=repeat, n_actions=len(action_history), actions=action_history)
        )
        result.update(metrics)
        results.append(result)
        pd.DataFrame(results).to_csv(f"{experiment_dir}/results.csv", index=False)
        print(results)

        with open(f"{repeat_dir}/prompt.txt", "w") as f:
            f.write(agent.prompt)

        # assert all(
        #     blicket_status[i] == BlicketStatus.yes for i in env._current_gt_hypothesis.blickets
        # )
        # assert all(
        #     blicket_status[i] == BlicketStatus.no
        #     for i in range(n_objects)
        #     if not blicket_status[i] == BlicketStatus.yes
        # )
        # print("All assertions passed. Solution is correct.")
