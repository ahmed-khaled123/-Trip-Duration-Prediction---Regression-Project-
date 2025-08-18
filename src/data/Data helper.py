"""
Helpers for loading and validating datasets.

Notes:
- اكتب هنا منطق تحميل البيانات من `data/raw/`.
- اعمل فحوصات: نطاق الإحداثيات، قيم ناقصة، أنواع الأعمدة.
- لا تلمس البيانات الخام: أي تعديل يكتب إلى `data/processed/`.
"""

def load_raw(path: str):
    """تحميل البيانات الخام من المسار المعطى (CSV/Parquet). أعد هيكلًا مناسبًا للتحليل."""
    pass

def sanity_checks(df):
    """تحقق أولي (missing/outliers/ranges). اكتب TODOs كبداية."""
    pass
