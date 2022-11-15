"""
Utility functions

"""
import numpy as np


def _enum(values, final="and"):
    """
    Generate an enumeration of words ending with some final word
    """
    values = list(values)
    if len(values) > 1:
        return f"{', '.join(values[: -1])} {final} {values[-1]}"
    else:
        return values[-1]


def _score_blickets(agent, true):
    print("Agent:", agent, "  true:", true)

    # Convert indices of blickets to binary array
    tmp = np.zeros(
        agent.shape[0],
    )
    tmp[list(true)] = 1
    true = tmp.astype(bool)
    del tmp

    metrics = {}
    metrics["accuracy"] = (agent == true).sum() / len(agent)
    metrics["tp"] = (agent == true)[true].sum()
    metrics["tn"] = (agent == true)[~true].sum()
    metrics["fp"] = (~true).sum() - metrics["tn"]
    metrics["fn"] = true.sum() - metrics["tp"]
    metrics["precision"] = metrics["tp"] / agent.sum()
    metrics["recall"] = metrics["tp"] / true.sum()
    metrics["f1"] = (
        2
        * metrics["precision"]
        * metrics["recall"]
        / (metrics["precision"] + metrics["recall"])
    )

    return metrics


def _verbalize_action(action, objects):
    """
    Convert an action array to its natural language specification

    Parameters:
    -----------
    action: np.ndarray, shape=(n_objects,)
        A binary array with one value per object, which is 1 if the object is placed on the detector and 0 otherwise.

    Returns:
    --------
    action_str: str
        The natural language specification of the action.

    """
    if action is None:
        return "Stop."
    elif sum(action) == 0:
        return "Do not place anything on the detector."
    else:
        tmp = _enum(
            [f"the {o}" for i, o in enumerate(objects) if action[i] == 1],
            final="and",
        )
        return f"Place {tmp} on the detector."
