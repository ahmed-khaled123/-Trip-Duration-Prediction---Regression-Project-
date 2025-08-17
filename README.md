# Trip Duration Prediction

هدف المشروع: التنبؤ بمدة الرحلة (بالثواني) اعتمادًا على معطيات مثل الإحداثيات ووقت الانطلاق وعدد الركاب… إلخ.

## لماذا الهيكلة دي؟
- تفصل بين البيانات الخام والمعالجة والكود.
- تسهّل الشغل على VS Code وجيت من أول يوم.
- تخلي التطوير خطوة بخطوة وواضح (EDA → Feature Engineering → Modeling).

## هيكلة المجلدات
```
trip-duration-prediction/
├── config/
│   └── params.yaml                # إعدادات عامة/بارامترات (تملأها لاحقًا)
├── data/
│   ├── raw/                       # البيانات الخام (لا تُعدّل)
│   ├── processed/                 # بيانات بعد التنظيف/الاشتقاق
│   └── external/                  # أي بيانات إضافية (طقس/خرائط…)
├── notebooks/
│   └── 01_eda_trip_duration.ipynb # نوتبوك الاستكشاف الأول
├── src/
│   ├── data/                      # كود التحميل/التقسيم/التحقق من البيانات
│   ├── features/                  # توليد الميزات (المسافات/الوقت…)
│   ├── models/                    # تدريب وتقييم النماذج
│   └── visualization/             # دوال مساعدة للرسم أثناء الـEDA
├── scripts/                       # سكربتات تشغيل سريعة (اختياري)
├── tests/                         # اختبارات بسيطة لوظائفك
├── .gitignore
├── Makefile
├── requirements.txt
└── README.md
```

## خطوات البدء السريع (VS Code + Git)
1) **إنشاء بيئة عمل**:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) **بدء EDA**:
- افتح VS Code في مجلد المشروع.
- افتح `notebooks/01_eda_trip_duration.ipynb` وامشي على الاهداف/الـTODOs.
- خليك ماشي على مبدأ: أسئلة → رسوم/جداول → استنتاجات قصيرة.

3) **تتبع بالإصدار (GitHub)**:
```bash
git init
git add .
git commit -m "chore: init trip-duration-prediction skeleton"
# انشئ Repo على GitHub ثم اربطه:
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```
- اشتغل دائمًا على فروع مميزة للمهام:
```bash
git checkout -b feature/eda-basic
# شغل… ثم:
git add -A
git commit -m "feat(eda): overview & sanity checks"
git push -u origin feature/eda-basic
# افتح Pull Request وادمجه بعد المراجعة الذاتية.
```

## سياسة البيانات
- **لا ترفع** ملفات كبيرة أو البيانات الخام للعامة (ضعها في `data/raw/` وموجودة في .gitignore).
- ارفع فقط الأكواد والملفات النصية المفيدة للتتبع.

## الخطوات القادمة (Checklist)
- [ ] وضع البيانات الخام داخل `data/raw/`
- [ ] فتح النوتبوك `01_eda_trip_duration.ipynb`
- [ ] كتابة قائمة أسئلة الـEDA قبل التنفيذ
- [ ] تنفيذ فحوصات السلامة للبيانات (قيم ناقصة/قيم شاذة/نطاق الإحداثيات)
- [ ] توليد ميزات أولية (المسافات، الوقت) **بدون نموذج** كبداية
- [ ] توثيق الاستنتاجات بنقاط موجزة أسفل كل قسم
