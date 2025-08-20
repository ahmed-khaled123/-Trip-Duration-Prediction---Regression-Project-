def split_features_target(train_df, val_df, target_col):
    #split data into features and target
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    return X_train, y_train, X_val, y_val

