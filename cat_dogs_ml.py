import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# ==============================================================================
# AYARLAR (YENİ KLASÖR İSMİNE GÖRE GÜNCELLENDİ)
# ==============================================================================
# "makine_ogrenme" klasörüne göre yolları sabitliyoruz.
TRAIN_PATH = r"C:\Users\user\Desktop\makine_ogrenme\training_set"
TEST_PATH = r"C:\Users\user\Desktop\makine_ogrenme\test_set"

CATEGORIES = ['cats', 'dogs']
IMG_SIZE = 64  # Resim boyutu


def load_data_cnn(data_path):
    data = []
    labels = []

    print(f"\n--> '{data_path}' klasörü taranıyor...")

    # Klasör kontrolü
    if not os.path.exists(data_path):
        print(f"HATA: Klasör bulunamadı!\nAranan Yol: {data_path}")
        return np.array([]), np.array([])

    for category in CATEGORIES:
        path = os.path.join(data_path, category)
        class_num = CATEGORIES.index(category)  # 0: kedi, 1: köpek

        if not os.path.exists(path):
            print(f"   [UYARI] '{category}' klasörü yok! ({path})")
            continue

        print(f"   - '{category}' yükleniyor...")
        count = 0
        for img_name in os.listdir(path):
            try:
                # Sadece resim dosyalarını al (.jpg, .png)
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                img_path = os.path.join(path, img_name)
                # Siyah beyaz okuma (Gri tonlama)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img_array is None:
                    continue

                # Boyutlandırma (64x64)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append(new_array)
                labels.append(class_num)
                count += 1
            except Exception as e:
                pass
        print(f"     -> {count} resim alındı.")

    X = np.array(data)
    y = np.array(labels)

    # Veri varsa normalize et ve şekillendir
    if len(X) > 0:
        X = X / 255.0
        # (Adet, 64, 64, 1) formatına çevir
        X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return X, y


def cnn_model_kur():
    """
    Basit CNN Modeli Mimarisi
    """
    model = Sequential()

    # 1. Konvolüsyon Katmanı
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D((2, 2)))

    # 2. Konvolüsyon Katmanı
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # 3. Düzleştirme ve Tahmin
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 0 veya 1 çıktısı verir

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def tekli_tahmin_cnn(model):
    print("\n" + "=" * 30)
    print("   TEKLİ TAHMİN MODU")
    print("   Çıkmak için 'q' yaz.")
    print("=" * 30)

    while True:
        dosya_adi = input("\nResim adını girin (Örn: dog.2): ")
        if dosya_adi.lower() == 'q':
            print("Çıkış yapılıyor...")
            break

        # Senin resimlerin .jpg uzantılı, otomatik ekliyoruz
        tam_dosya_adi = dosya_adi + ".jpg"

        # Resmi bulmak için tam yollara bakıyoruz
        resim_yolu = None

        # Önce dogs, sonra cats klasöründe ara
        path_dogs = os.path.join(TRAIN_PATH, 'dogs', tam_dosya_adi)
        path_cats = os.path.join(TRAIN_PATH, 'cats', tam_dosya_adi)

        if os.path.isfile(path_dogs):
            resim_yolu = path_dogs
        elif os.path.isfile(path_cats):
            resim_yolu = path_cats

        if not resim_yolu:
            print(f"HATA: '{tam_dosya_adi}' bulunamadı!")
            print(f"Kontrol edilen yer: {TRAIN_PATH}")
            continue

        try:
            # Tahmin işlemi
            img_array = cv2.imread(resim_yolu, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            # Modele uygun hale getir
            hazir_veri = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            hazir_veri = hazir_veri / 255.0

            sonuc = model.predict(hazir_veri)
            skor = sonuc[0][0]

            print(f"Model Skoru: {skor:.4f}")
            if skor > 0.5:
                print(f"--> Tahmin: KÖPEK 🐶")
            else:
                print(f"--> Tahmin: KEDİ 🐱")

        except Exception as e:
            print(f"Bir hata oluştu: {e}")


def main():
    print("--- 1. Veriler Yükleniyor ---")
    X_train, y_train = load_data_cnn(TRAIN_PATH)

    if len(X_train) == 0:
        print("\n!!! KRİTİK HATA !!!")
        print("Hala resim bulunamadıysa, lütfen masaüstündeki klasörün içine girip")
        print("'training_set' adında bir klasör olduğunu teyit et.")
        return

    X_test, y_test = load_data_cnn(TEST_PATH)

    print("\n--- 2. Model Kuruluyor ve Eğitiliyor ---")
    model = cnn_model_kur()

    # Epoch (Tur) eğitim yapacağız
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    print("\n--- 3. Sonuçlar ---")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Model Doğruluğu (Accuracy): %{acc * 100:.2f}")

    # Tahmin modunu başlat
    tekli_tahmin_cnn(model)


if __name__ == "__main__":
    main()