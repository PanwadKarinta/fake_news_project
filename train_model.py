import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# โหลดข้อมูล
true_df = pd.read_csv("true_df_cleaned.csv")
fake_df = pd.read_csv("fake_df_cleaned.csv")

# เพิ่ม label
true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

# รวมข้อมูล
data = pd.concat([true_df, fake_df])

# ใช้ column text
X = data["text"]
y = data["label"]

# แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 🔥 แนะนำเพิ่ม เพื่อ balance class
)

# แปลงข้อความเป็น TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# สร้างโมเดล
model = RandomForestClassifier(n_estimators=300, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    max_depth=None, 
    bootstrap=False,
    random_state=42)
model.fit(X_train_tfidf, y_train)

# ประเมินผล
y_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", test_accuracy)

# 🔥 บันทึกทุกอย่าง
joblib.dump(model, "best_random_forest_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(test_accuracy, "model_accuracy.pkl")  # ✅ เพิ่มบรรทัดนี้

print("Model training complete and files saved!")