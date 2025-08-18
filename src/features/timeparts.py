"""
Feature engineering: time-based features from pickup/dropoff datetimes.

- استخرج hour/weekday/month/is_weekend/is_rush_hour.
- راقب المناطق الزمنية والتواريخ الغلط.
"""

"""
Feature engineering: time-based features from pickup/dropoff datetimes.

- استخرج hour/weekday/month/is_weekend/is_rush_hour.
- راقب المناطق الزمنية والتواريخ الغلط.
"""

import pandas as pd
"""
Feature engineering: time-based features from pickup/dropoff datetimes.

- استخرج hour/weekday/month/is_weekend/is_rush_hour.
- راقب المناطق الزمنية والتواريخ الغلط.
"""



def add_time_features(df, datetime_col="pickup_datetime"):
    """
    Extract time-based features from a datetime column:
    - hour
    - weekday
    - month
    - is_weekend
    - is_rush_hour (based on actual data distribution: morning 8-10, evening 17-21)
    """
    # تحويل النص لتاريخ/وقت لو مش datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # ساعة البداية
    df['hour'] = df[datetime_col].dt.hour
    
    # يوم الأسبوع
    df['weekday'] = df[datetime_col].dt.weekday
    
    # الشهر
    df['month'] = df[datetime_col].dt.month
    
    # weekend flag
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
    
    # rush hour flag (actual data distribution)
    df['is_rush_hour'] = ((df['hour'].between(8,10)) | (df['hour'].between(17,21))).astype(int)
    
    # مراقبة أي قيم ناقصة
    missing_dates = df[datetime_col].isnull().sum()
    if missing_dates > 0:
        print(f"Warning: {missing_dates} invalid datetime values in {datetime_col} converted to NaT")
    
    return df
