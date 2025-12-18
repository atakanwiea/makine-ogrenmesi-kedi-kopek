#!/usr/bin/env python3
"""
svm_raw_pixels.py

Ham piksel (64x64 grayscale -> 4096 boyut) kullanarak SVM ile kedi/köpek sınıflandırması.
Kullanım:
    python svm_raw_pixels.py

Ayarlar dosyanın içinde (TRAIN_PATH, TEST_PATH, IMG_SIZE).
Gerekli paketler:
    pip install numpy opencv-python scikit-learn joblib
"""
import os
import cv2
import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings

# Ayarlar — öncelikle sabit, mutlak yollara bak (VS Code cwd sorunlarını önlemek için)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Kullanıcının verdiği kesin yollar
TRAIN_PATH = r"C:\Users\ataka\Desktop\MachineLearningProject\training_set\training_set"
TEST_PATH = r"C:\Users\ataka\Desktop\MachineLearningProject\test_set\test_set"
# Eğer sabit yollar bulunamazsa repo-ilişkili yollara geri dön
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = os.path.join(BASE_DIR, 'training_set', 'training_set')
if not os.path.exists(TEST_PATH):
    TEST_PATH = os.path.join(BASE_DIR, 'test_set', 'test_set')
CATEGORIES = ['cats', 'dogs']
IMG_SIZE = 64  # (cat_dogs_ml.py ile aynı)
USE_PCA = False  # Ham piksel isteğe bağlı PCA ile boyut azaltma (default False)

def load_images_as_flat(data_path):
    X = []
    y = []
    for idx, cat in enumerate(CATEGORIES):
        folder = os.path.join(data_path, cat)
        if not os.path.exists(folder):
            print(f"[UYARI] Klasör bulunamadı: {folder}")
            continue
        filenames = os.listdir(folder)
        count = 0
        for fname in filenames:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            p = os.path.join(folder, fname)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            vec = img.flatten()  # 64*64 = 4096
            X.append(vec)
            y.append(idx)
            count += 1
        print(f"  - {cat}: {count} resim yüklendi.")
    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def main():
    print("Veriler yükleniyor...")
    print(f"Kullanılacak TRAIN_PATH: {TRAIN_PATH}")
    print(f"Kullanılacak TEST_PATH: {TEST_PATH}")
    X_train, y_train = load_images_as_flat(TRAIN_PATH)
    X_test, y_test = load_images_as_flat(TEST_PATH)

    if X_train.size == 0:
        print("Eğitim verisi bulunamadı. TRAIN_PATH'i kontrol edin.")
        return
    if X_test.size == 0:
        print("Test verisi bulunamadı. TEST_PATH'i kontrol edin.")
        return

    print(f"Eğitim örnek sayısı: {len(y_train)}, Test örnek sayısı: {len(y_test)}")
    # Ölçekleme (çok önemli)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # İsteğe bağlı PCA ile boyut azaltma
    if USE_PCA:
        print("PCA uygulanıyor (örnek: 0.98 varyans koruma)...")
        pca = PCA(n_components=0.98, svd_solver='full')
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s = pca.transform(X_test_s)
    else:
        pca = None

    # Çapraz doğrulama katman sayısını veriye göre belirle (en az 2 olmalı)
    class_counts = Counter(y_train)
    min_count = min(class_counts.values()) if class_counts else 0
    cv = min(5, min_count) if min_count >= 2 else 2

    # Eğer veri çok çok küçükse GridSearch yerine direkt eğitim yap
    perform_grid = cv >= 2 and len(y_train) >= cv * 2

    # Model / GridSearch
    if perform_grid:
        print(f"GridSearch ile LinearSVC optimizasyonu (cv={cv}) başlatılıyor...")
        param_grid = {'C': [0.01, 0.1, 1, 10]}
        svc = LinearSVC(max_iter=5000, dual=False)
        # Use single-threaded GridSearch on Windows to avoid loky/worker access violations
        grid = GridSearchCV(svc, param_grid, cv=cv, n_jobs=1, verbose=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid.fit(X_train_s, y_train)
        best = grid.best_estimator_
        print("En iyi parametreler:", grid.best_params_)
    else:
        print("GridSearch yapılamıyor (veri çok küçük). Varsayılan LinearSVC ile eğitim yapılıyor.")
        best = LinearSVC(max_iter=5000, dual=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best.fit(X_train_s, y_train)

    # Değerlendirme
    print("Test setinde tahmin yapılıyor...")
    try:
        y_pred = best.predict(X_test_s)
    except Exception as e:
        print("Tahmin sırasında hata:", e)
        return

    acc = accuracy_score(y_test, y_pred)
    print(f"Test doğruluk (accuracy): {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Modeli kaydet
    model_bundle = {
        'model': best,
        'scaler': scaler,
        'pca': pca,
        'img_size': IMG_SIZE,
        'categories': CATEGORIES
    }
    joblib.dump(model_bundle, "svm_raw_pixels_model.pkl")
    print("Model kaydedildi: svm_raw_pixels_model.pkl")

if __name__ == "__main__":
    main()