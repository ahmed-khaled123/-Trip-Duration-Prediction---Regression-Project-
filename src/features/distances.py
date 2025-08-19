"""
Feature engineering: distances & bearings

فكر بصوت عالي (داخل التعليقات):
- هستخدم أي صيغة للمسافة؟ Haversine/Manhattan.
- هل في تحجيم للوحدات؟
- إيه الـ edge cases (إحداثيات صفر/قيم ناقصة)؟
"""

"""
Feature engineering: distances & bearings

فكر بصوت عالي (داخل التعليقات):
- هستخدم أي صيغة للمسافة؟ Haversine/Manhattan.
  → ممكن نضيف الاثنين: Haversine للبعد الواقعي على سطح الأرض،
    Manhattan عشان تقريبي للشوارع في المدينة.
- هل في تحجيم للوحدات؟
  → الدوال بتطلع المسافة بالكيلومتر، ممكن نعمل scaling لاحقًا لو النموذج محتاج.
- إيه الـ edge cases (إحداثيات صفر/قيم ناقصة)؟
  → لو خط العرض أو الطول ناقص أو صفر → نرجع NaN أو نتجاهل.
    كمان لازم نتأكد إن الإحداثيات ضمن نطاق منطقي (مثلاً NYC: lat 40.5-41, lon -74 to -73)
"""

import numpy as np
import pandas as pd

def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distance-based features: haversine, manhattan, bearing, speed.
    """
    # =============================
    # Haversine distance
    # =============================
    def haversine_array(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km
    

    df["haversine_km"] = haversine_array(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"]
    )

    # Manhattan approximation
    df["manhattan_km"] = (
        haversine_array(df["pickup_longitude"], df["pickup_latitude"],
                        df["pickup_longitude"], df["dropoff_latitude"]) +
        haversine_array(df["pickup_longitude"], df["dropoff_latitude"],
                        df["dropoff_longitude"], df["dropoff_latitude"])
    )

    # Bearing
    def bearing_array(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return np.degrees(np.arctan2(y, x))

    df["bearing_deg"] = bearing_array(
        df["pickup_longitude"], df["pickup_latitude"],
        df["dropoff_longitude"], df["dropoff_latitude"]
    )

    # =============================
    # Speed calculation (km/h)
    # =============================
    df["speed_kmh"] = np.nan
    mask = (df["trip_duration"] > 0) & df["haversine_km"].notnull()
    df.loc[mask, "speed_kmh"] = (
        df.loc[mask, "haversine_km"] / (df.loc[mask, "trip_duration"] / 3600)
    )

    # Debug info
    total_rows = len(df)
    valid_rows = mask.sum()
    print(f"✅ add_distance_features: {valid_rows}/{total_rows} rows got valid speed_kmh")

    return df
