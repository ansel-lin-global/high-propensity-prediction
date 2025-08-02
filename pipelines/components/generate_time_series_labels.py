from typing import Annotated
from kfp.dsl import component, Dataset, Input, Output

@component(
    base_image="python:3.10",
    packages_to_install=["pandas"]
)
def generate_time_series_labels(
    input_data: Input[Dataset],
    output_labeled_data: Output[Dataset],
    user_id_col: str,
    timestamp_col: str,
    event_col: str,
    observation_days: int,
    prediction_days: int
):
    """
    Build rolling time-series windows for supervised learning.

    For each user:
    - Use a 10-day observation window to extract behavior features
    - Follow with a 3-day prediction window to assign a binary label
    - Label = 1 if target event (e.g. purchase) occurs in prediction window

    This approach prevents data leakage and enables generalization across time.
    """
    import pandas as pd
    from datetime import timedelta

    df = pd.read_parquet(input_data.path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([user_id_col, timestamp_col])

    samples = []

    for user_id, user_df in df.groupby(user_id_col):
        user_df = user_df.reset_index(drop=True)

        for i in range(len(user_df)):
            obs_start = user_df.loc[i, timestamp_col]
            obs_end = obs_start + timedelta(days=observation_days)
            pred_end = obs_end + timedelta(days=prediction_days)

            obs_window = user_df[(user_df[timestamp_col] >= obs_start) & 
                                 (user_df[timestamp_col] < obs_end)]
            pred_window = user_df[(user_df[timestamp_col] >= obs_end) & 
                                  (user_df[timestamp_col] < pred_end)]

            label = int((pred_window[event_col] == 'purchase').any())

            samples.append({
                "user_id": user_id,
                "obs_start": obs_start,
                "obs_end": obs_end,
                "label": label,
                # Placeholder for features
            })

    labeled_df = pd.DataFrame(samples)
    labeled_df.to_parquet(output_labeled_data.path, index=False)
