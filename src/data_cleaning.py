"""DATA CLEANING & GEOMETRIC RECONSTRUCTION MODULE

This file handles the messy, physically impossible values in the raw dataset.
Diamonds exist in 3D space, so we use real gemology formulas to fix errors
rather than guessing with medians or throwing data away.
"""

import pandas as pd


def clean_diamond_data(df: pd.DataFrame) -> pd.DataFrame:
    """Takes in the raw diamonds DataFrame and returns a clean DataFrame

    grounded in real-world physical dimensions.
    """

    # THOUGHT PROCESS:
    # Always make an explicit copy so we don't accidentally modify the
    # original raw dataset in memory.
    df_clean = df.copy()

    # -------------------------------------------------------------------------
    # STEP 1: REMOVE PHYSICALLY IMPOSSIBLE BASE MEASUREMENTS (x and y)
    # -------------------------------------------------------------------------
    # THOUGHT PROCESS & PHYSICAL REALITY:
    # x = length (mm), y = width (mm).
    # A real physical solid object cannot have a length or width of 0 mm.
    # If a diamond has no length or width, it has no base footprint, making it
    # mathematically impossible to reconstruct its volume or depth.
    # Therefore, these corrupted rows must be safely removed.
    df_clean = df_clean[df_clean["x"] > 0]
    df_clean = df_clean[df_clean["y"] > 0]

    # -------------------------------------------------------------------------
    # STEP 2: RECONSTRUCT MISSING DEPTH (z == 0) USING GEMOLOGY MATH
    # -------------------------------------------------------------------------
    # THOUGHT PROCESS & MATH FORMULA:
    # z = depth (mm) from the top table down to the culet point.
    # Rather than throwing away diamonds that only have z == 0, we can use the
    # formula diamond cutters use:
    #
    #   depth percentage = ( z / ((x + y) / 2) ) * 100
    #
    # We rearranged this formula with algebra to solve for missing z:
    #   1. depth / 100 = z / ((x + y) / 2)
    #   2. z = (depth / 100) * ((x + y) / 2)
    #   3. z = depth * (x + y) / 200
    #
    # This recalculates the true depth using the stone's own length and width!

    # Create a True/False mask to find only the diamonds missing depth:
    missing_z = df_clean["z"] == 0

    # Calculate and assign z using the formula only where missing_z is True:
    df_clean.loc[missing_z, "z"] = (
        df_clean["depth"] * (df_clean["x"] + df_clean["y"]) / 200
    )

    # Return the clean, physically valid dataset ready for modeling
    return df_clean
