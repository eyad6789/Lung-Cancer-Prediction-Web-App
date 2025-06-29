# import joblib

# # 1) حمّل الموديل والـ scaler
# model  = joblib.load('best_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # 2) مثال: شخص خالي من الأعراض (GENDER=0, AGE=25, كل البقية=0)
# features = [[0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# # 3) طبق الـ scaler مباشرة على الـ list-of-lists
# scaled = scaler.transform(features)  # => numpy array

# # 4) استخرج الاحتمالات وال prediction
# probs = model.predict_proba(scaled)[0]
# pred  = model.predict(scaled)[0]

# print(f"Prob(no cancer) = {probs[0]:.4f}, Prob(cancer) = {probs[1]:.4f}")
# print("Prediction class:", pred, "(0 = no cancer, 1 = cancer)")

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# 1) حمّل البيانات (نفس preprocessing):
df = pd.read_csv("/content/survey-lung-cancer.csv")
df['GENDER']       = (df['GENDER']=='M').astype(int)
df['LUNG_CANCER']  = (df['LUNG_CANCER']=='YES').astype(int)

# 2) اختر نفس الـ features وترتيبها:
cols = [
  'GENDER','AGE','SMOKING','YELLOW FINGERS','ANXIETY','PEER PRESSURE',
  'CHRONIC DISEASE','FATIGUE','ALLERGY','WHEEZING','ALCOHOL CONSUMING',
  'COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN'
]
X = df[cols]
y = df['LUNG_CANCER']

# 3) جهّز scaler والموديل
scaler = joblib.load('scaler.pkl')
model  = joblib.load('best_model.pkl')

# 4) حضّر مجموعة الاختبار (مثلاً استخدم 30% كما في تدريبك)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_test_scaled = scaler.transform(X_test)
y_pred       = model.predict(X_test_scaled)

# 5) طبع تقارير الأداء
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['No Cancer','Cancer']))
