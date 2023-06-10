import numpy as np
import pandas as pd



def postprocess_joint(joint_dist):
    """ casts joint distribution to a dataframe, and appends the predicted maneuver
    """
    joint_df = pd.DataFrame({
            'takeoff': joint_dist[:, 0],
            'turn': joint_dist[:, 1],
            'line': joint_dist[:, 2],
            'orbit': joint_dist[:, 3],
            'landing': joint_dist[:, 4]
        })
    joint_df['maneuver'] = joint_df.idxmax(axis="columns")
    return joint_df


