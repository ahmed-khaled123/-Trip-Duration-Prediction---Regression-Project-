import pandas as pd


def add_one_hot_encoding(df: pd.DataFrame, categorical_cols=None) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical features.
    """
    if categorical_cols is None:
        categorical_cols = ["vendor_id", "store_and_fwd_flag", "weekday", "month"]

    # هنا drop_first بياخد قيمة واحدة (True أو False) مش ليستة
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)


    return df_encoded