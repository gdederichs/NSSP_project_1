import numpy as np
import pandas as pd


def reg_motion(motion_outliers, n_scans):
    new_regs = []
    new_reg_names = []
    for idx, out_frame in enumerate(motion_outliers):
        column_values = np.zeros(n_scans)
        column_values[out_frame] = 1
        new_regs.append(column_values)
        new_reg_names.append(f"motion_outlier_{idx}")
    return np.vstack(new_regs), new_reg_names
    
def condition_vector(position: int, n_regressors: int) -> np.array:
        vec = np.zeros((1, n_regressors))
        vec[0, position] = 1
        return vec
    
def get_conditions_of_interest(design: pd.DataFrame, keys_to_keep: list) -> dict:
    """
    Creates a dictionary of conditions of interest based on specified keys to keep.

    Parameters:
    - design (pd.DataFrame): Design matrix with columns representing regressors.
    - keys_to_keep (list): List of condition names to keep.

    Returns:
    - dict: A dictionary where keys are condition names and values are condition vectors.
    """
    
    n_regressors = design.shape[1]
    conditions = {col: condition_vector(idx, n_regressors) for idx, col in enumerate(design.columns)}
    conditions_of_interest = {key: conditions[key] for key in keys_to_keep if key in conditions}
    
    return conditions_of_interest