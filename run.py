import numpy as np

from collections import defaultdict
from enum import Enum
from scipy.special import softmax

from causal_env_v0 import *
from agents import GPT3Agent


class BlicketStatus(Enum):
    unknown = 0
    maybe = 1
    yes = 2
    no = 3


def name_objects(n_objects):
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

    from itertools import product

    # The following ensures unique color and shape for each object.
    return [
        f"{c} {s}"
        for c, s in zip(
            np.random.choice(colors, size=n_objects, replace=False),
            np.random.choice(shapes, size=n_objects, replace=False),
        )
    ]

    # combos = [f"{c} {k}" for k, c in product(shapes, colors)]
    # return list(np.random.choice(combos, size=n_objects, replace=False))


n_objects = 3

env = CausalEnv_v0(
    {
        "reward_structure": "baseline",
        "hypotheses": [
            ABconj
        ],  # We fully control the hypothesis (no sampling among a list)
        "n_blickets": n_objects,
    }
)

env_type = (
    "conjunctive"
    if issubclass(env._current_gt_hypothesis, ConjunctiveHypothesis)
    else "disjunctive"
)


for repeat in range(1):
    # We have a fixed environment, but the object names change.

    objects = name_objects(n_objects)
    print("The objects are:", ", ".join(objects))
    print()
    print(
        "Ground truth: the blickets are the",
        ", ".join([objects[i] for i in env._current_gt_hypothesis.blickets]),
        env._current_gt_hypothesis,
        f"and the detector is {'conjunctive' if issubclass(env._current_gt_hypothesis, ConjunctiveHypothesis) else 'disjunctive'}.",
    )
    print()

    # Instantiate the agent
    # TODO: here we assume conjunction and tell the agent that it is.
    agent = GPT3Agent(n_objects=n_objects, objects=objects, random_actions=False)
    action_history = []
    # agent = RandomAgent(n_objects=n_objects)

    # Memory to keep track of what we know
    blicket_status = defaultdict(lambda: BlicketStatus.unknown)
    is_answerable = False  # Track if we have gathered enough info to make a call on each blicket (in theory)

    # Define the initial state of the system (shown as an experiment to the agent)
    # TODO: make starting state an option.
    action = np.ones(n_objects)
    detector_activated = True

    for j in range(10):
        if all(
            blicket_status[idx] in [BlicketStatus.yes, BlicketStatus.no]
            for idx in range(n_objects)
        ):
            is_answerable = True

        action = agent.next_action(
            state_info={"prev_action": action, "detector_activated": detector_activated}
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
            print("|_________ Reward: The detector did turn on (DETECTOR ACTIVATED).")
        else:
            print("|_________ Reward: The detector did not turn on.")

        # Update our knowledge of blick statuses
        if env_type == "conjunctive":
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

        print(
            "|_________ Blicket status:",
            [blicket_status[idx] for idx in range(n_objects)],
        )

    print("Exploration complete. Asking agent to decide blickets.")
    print(action_history)
    print(agent.decide_blickets())

    # assert all(
    #     blicket_status[i] == BlicketStatus.yes for i in env._current_gt_hypothesis.blickets
    # )
    # assert all(
    #     blicket_status[i] == BlicketStatus.no
    #     for i in range(n_objects)
    #     if not blicket_status[i] == BlicketStatus.yes
    # )
    # print("All assertions passed. Solution is correct.")
