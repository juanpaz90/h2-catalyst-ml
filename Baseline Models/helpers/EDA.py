    # find general duplicates
import pandas as pd
import numpy as np

def check_duplicates_with_tensors_iterative(df):
    """
    Optimized function to check for duplicates in a DataFrame, handling nested numpy.ndarray (e.g., tensors)
    and scalar values like numpy.float32, without using recursion.

    Parameters:
        df (pd.DataFrame): The DataFrame to check for duplicates.

    Returns:
        int: The number of duplicate rows in the DataFrame.
    """
    def convert_to_hashable(value):
        """
        Convert numpy.ndarray or nested structures to hashable types (e.g., tuples) iteratively.
        Leave scalar values (e.g., numpy.float32) unchanged.
        """
        if isinstance(value, np.ndarray):
            # Flatten the array and convert to a tuple
            return tuple(value.ravel())
        elif isinstance(value, (list, tuple)):
            # Convert lists or tuples to tuples of hashable elements
            return tuple(convert_to_hashable(v) for v in value)
        else:
            # Leave scalar values unchanged
            return value

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Convert unhashable columns (e.g., numpy.ndarray) to hashable types
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(convert_to_hashable)

    # Check for duplicates
    duplicate_count = df_copy.duplicated().sum()

    print(f"Total duplicate rows: {duplicate_count}")
    return duplicate_count