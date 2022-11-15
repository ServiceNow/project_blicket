"""
Utility functions

"""


def _enum(values, final="and"):
    """
    Generate an enumeration of words ending with some final word
    """
    values = list(values)
    if len(values) > 1:
        return f"{', '.join(values[: -1])} {final} {values[-1]}"
    else:
        return values[-1]


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
