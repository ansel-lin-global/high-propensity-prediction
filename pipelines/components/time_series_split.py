import pandas as pd
from datetime import timedelta

def generate_time_series_labels(df: pd.DataFrame, 
                                 user_id_col: str, 
                                 timestamp_col: str,
                                 event_col: str,
                                 prediction_window_days: int = 3,
                                 observation_window_days: int = 14) -> pd.DataFrame:
    """
    Construct rolling windows from event log for supervised learning.

    Args:
        df (pd.DataFrame): Raw GA4 event log.
        user_id_col (str): Column with unique user ID.
        timestamp_col (str): Timestamp of event.
        event_col (str): Event type (e.g., 'add_to_cart', 'purchase').
        prediction_window_days (int): Days after observation window to label positive samples.
        observation_window_days (int): Days to observe user behavior.

    Returns:
        pd.DataFrame: Sampled windows with binary label.
    """

    # Make sure timestamp is in datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Sort for rolling window logic
    df = df.sort_values(by=[user_id_col, timestamp_col])

    samples = []

    for user_id, user_df in df.groupby(user_id_col):
        user_df = user_df.reset_index(drop=True)

        for i in range(len(user_df)):
            obs_start = user_df.loc[i, timestamp_col]
            obs_end = obs_start + timedelta(days=observation_window_days)
            pred_end = obs_end + timedelta(days=prediction_window_days)

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
                # Placeholder — in practice you'd include aggregated features
                "event_count": len(obs_window)
            })

    return pd.DataFrame(samples)
