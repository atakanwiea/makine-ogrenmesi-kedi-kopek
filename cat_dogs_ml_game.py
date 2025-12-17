import os
import cv2
import numpy as np
import random
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import math

# --- AYARLAR ---
# Öncelikle çalıştığımız script'in bulunduğu klasöre göre göreli eğitim yolunu kullan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'training_set', 'training_set')
# Eğer repo içindeki yol yoksa eski (dev) sabit yola geri dön
if not os.path.exists(TRAIN_PATH):
    ANA_KLASOR = r"C:\Users\user\Desktop\makine_ogrenme"
    TRAIN_PATH = os.path.join(ANA_KLASOR, 'training_set')

IMG_SIZE_ML = 64
GRID_SIZE = 4
GAME_SIZE = 800
PANEL_WIDTH = 250
CELL_SIZE = GAME_SIZE // GRID_SIZE


# --- MODEL KISMI ---
def load_data_simple(data_path):
    data, labels = [], []
    categories = ['cats', 'dogs']
    for category in categories:
        path = os.path.join(data_path, category)
        class_num = categories.index(category)
        if not os.path.exists(path): continue
        for img_name in os.listdir(path):
            try:
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')): continue
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE_ML, IMG_SIZE_ML))
                data.append(new_array)
                labels.append(class_num)
            except:
                pass
    X = np.array(data) / 255.0
    y = np.array(labels)
    if len(X) > 0: X = X.reshape(-1, IMG_SIZE_ML, IMG_SIZE_ML, 1)
    return X, y


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE_ML, IMG_SIZE_ML, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- OYUN MOTORU ---
class GameApp:
    def __init__(self, root, model, X_train, y_train, X_test, y_test):
        self.root = root
        self.model = model

        # VERİ SETLERİ (Hafıza Tazeleme için gerekli)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.root.title("AI Dashboard: Akıllı Hafıza Tazeleme Modu 🧠")
        self.root.geometry(f"{GAME_SIZE + PANEL_WIDTH}x{GAME_SIZE}")
        self.root.configure(bg="#1e1e1e")

        # Değişkenler
        self.score = 0
        self.total_moves = 0
        self.correct_moves = 0
        self.start_acc = self.calculate_model_accuracy()
        self.prev_acc = self.start_acc

        self.image_refs = [None] * (GRID_SIZE * GRID_SIZE)
        self.grid_data = [None] * (GRID_SIZE * GRID_SIZE)
        self.all_images_pool = self.get_all_image_paths()

        # Arayüz
        self.canvas = Canvas(root, width=GAME_SIZE, height=GAME_SIZE, bg="black", highlightthickness=0)
        self.canvas.pack(side="left")
        self.canvas.bind("<Button-1>", self.on_click)

        self.panel = tk.Frame(root, width=PANEL_WIDTH, height=GAME_SIZE, bg="#2d2d2d")
        self.panel.pack(side="right", fill="both", expand=True)
        self.panel.pack_propagate(False)

        self.create_dashboard_widgets()
        self.fill_initial_grid()

    def calculate_model_accuracy(self):
        if len(self.X_test) == 0: return 0.0
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return acc * 100

    def create_dashboard_widgets(self):
        tk.Label(self.panel, text="AKILLI ANALİZ", font=("Arial", 16, "bold"), bg="#2d2d2d", fg="#00ffcc",
                 pady=20).pack()

        # SKOR
        frame_score = tk.Frame(self.panel, bg="#333", pady=10, padx=10)
        frame_score.pack(fill="x", padx=10, pady=5)
        self.lbl_score = tk.Label(frame_score, text="PUAN: 0", font=("Consolas", 18, "bold"), bg="#333", fg="white")
        self.lbl_score.pack()

        # ACCURACY
        frame_acc = tk.Frame(self.panel, bg="#333", pady=15, padx=10)
        frame_acc.pack(fill="x", padx=10, pady=10)
        tk.Label(frame_acc, text="MODEL DOĞRULUĞU", font=("Arial", 10, "bold"), bg="#333", fg="#aaa").pack()
        self.lbl_acc_val = tk.Label(frame_acc, text=f"%{self.start_acc:.1f}", font=("Impact", 36), bg="#333",
                                    fg="#3399ff")
        self.lbl_acc_val.pack()
        self.lbl_acc_change = tk.Label(frame_acc, text="- Başlangıç -", font=("Arial", 12, "bold"), bg="#333",
                                       fg="gray")
        self.lbl_acc_change.pack()

        # LOGLAR
        tk.Label(self.panel, text="İşlem Kaydı:", font=("Arial", 10), bg="#2d2d2d", fg="gray", anchor="w").pack(
            fill="x", padx=10, pady=(20, 0))
        self.log_text = tk.Text(self.panel, height=12, bg="black", fg="#00ff00", font=("Consolas", 9), state="disabled")
        self.log_text.pack(fill="x", padx=10, pady=5)

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", "> " + message + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def get_all_image_paths(self):
        paths = []
        for cat in ['cats', 'dogs']:
            p = os.path.join(TRAIN_PATH, cat)
            if os.path.exists(p):
                for f in os.listdir(p):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        paths.append(os.path.join(p, f))
        return paths

    def fill_initial_grid(self):
        self.canvas.delete("all")
        if not self.all_images_pool: return
        for i in range(GRID_SIZE * GRID_SIZE):
            self.update_single_cell(i)

    def update_single_cell(self, index):
        if not self.all_images_pool: return
        new_img_path = random.choice(self.all_images_pool)

        row = index // GRID_SIZE
        col = index % GRID_SIZE
        x_pos = col * CELL_SIZE
        y_pos = row * CELL_SIZE

        pil_img = Image.open(new_img_path)
        pil_img = pil_img.resize((CELL_SIZE, CELL_SIZE))
        tk_img = ImageTk.PhotoImage(pil_img)

        if self.grid_data[index] and 'id' in self.grid_data[index]:
            self.canvas.delete(self.grid_data[index]['id'])

        img_id = self.canvas.create_image(x_pos, y_pos, anchor="nw", image=tk_img)
        self.image_refs[index] = tk_img
        self.grid_data[index] = {'path': new_img_path, 'x': x_pos, 'y': y_pos, 'id': img_id}

    def on_click(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        index = row * GRID_SIZE + col
        if 0 <= index < len(self.grid_data):
            data = self.grid_data[index]
            self.check_move(data['path'], data['x'], data['y'], index, event.x, event.y)

    def check_move(self, img_path, cell_x, cell_y, index, click_x, click_y):
        self.total_moves += 1

        # Resmi Hazırla
        try:
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE_ML, IMG_SIZE_ML))
            X_input = new_array.reshape(1, IMG_SIZE_ML, IMG_SIZE_ML, 1) / 255.0
        except:
            return

        # Tahmin
        prediction = self.model.predict(X_input, verbose=0)
        skor = prediction[0][0]
        model_tahmini = "KÖPEK" if skor > 0.5 else "KEDİ"

        # Gerçek
        dosya_ismi = os.path.basename(img_path).lower()
        gercek = "KÖPEK" if "dog" in dosya_ismi else "KEDİ"
        gercek_deger = 1 if gercek == "KÖPEK" else 0

        if model_tahmini == gercek:
            # DOĞRU
            self.score += 10
            self.create_particles(click_x, click_y, ["✓", "+10"], "#00ff00")
            self.log(f"Doğru bildi! ({model_tahmini})")
        else:
            # YANLIŞ -> GÜÇLENDİRİLMİŞ EĞİTİM (Hafıza Tazeleme)
            self.score -= 5
            self.log(f"HATA! {gercek} yerine {model_tahmini} dedi.")
            self.log("Hafıza tazelenerek eğitiliyor...")

            # --- STRATEJİ: REPLAY BUFFER ---
            # Sadece hatayı değil, rastgele 8 eski doğruyu da karıştırıp eğitiyoruz.

            # 1. Eğitim setinden rastgele 8 örnek seç
            random_indices = np.random.choice(len(self.X_train), 8, replace=False)
            X_replay = self.X_train[random_indices]
            y_replay = self.y_train[random_indices]

            # 2. Hatalı örneği bunların arasına kat
            X_batch = np.vstack([X_replay, X_input])
            y_batch = np.append(y_replay, gercek_deger)

            # 3. Hepsini birlikte eğit (Batch Training)
            # Böylece model "sadece bunu öğren" demez, "bunu öğren ama eskileri de hatırla" der.
            self.model.fit(X_batch, y_batch, epochs=5, verbose=0)

            self.create_particles(click_x, click_y, ["Düzeltildi", "♻️"], "orange")
            self.canvas.create_rectangle(cell_x, cell_y, cell_x + CELL_SIZE, cell_y + CELL_SIZE, outline="orange",
                                         width=4)

        # İstatistik Güncelle
        self.update_stats()
        self.lbl_score.config(text=f"PUAN: {self.score}")
        self.root.after(400, lambda: self.update_single_cell(index))

    def update_stats(self):
        new_acc = self.calculate_model_accuracy()
        diff = new_acc - self.prev_acc
        self.prev_acc = new_acc

        acc_color = "#3399ff" if new_acc >= 80 else "#ffcc00"
        self.lbl_acc_val.config(text=f"%{new_acc:.1f}", fg=acc_color)

        if diff > 0.01:
            self.lbl_acc_change.config(text=f"⬆ +%{diff:.2f}", fg="#00ff00")
        elif diff < -0.01:
            self.lbl_acc_change.config(text=f"⬇ %{diff:.2f}", fg="#ff4444")
        else:
            self.lbl_acc_change.config(text="• Sabit", fg="gray")

    def create_particles(self, x, y, symbols, color):
        particles = []
        for _ in range(8):
            txt = random.choice(symbols)
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            p_id = self.canvas.create_text(x, y, text=txt, font=("Arial", 16, "bold"), fill=color)
            particles.append({'id': p_id, 'dx': math.cos(angle) * speed, 'dy': math.sin(angle) * speed, 'life': 15})
        self.animate_particles(particles)

    def animate_particles(self, particles):
        if not particles: return
        remaining = []
        for p in particles:
            self.canvas.move(p['id'], p['dx'], p['dy'])
            p['life'] -= 1
            if p['life'] > 0:
                remaining.append(p)
            else:
                self.canvas.delete(p['id'])
        if remaining:
            self.root.after(30, lambda: self.animate_particles(remaining))


if __name__ == "__main__":
    print("--- SİSTEM HAZIRLANIYOR ---")
    print(f"Aranacak eğitim klasörü: {TRAIN_PATH}")
    X_all, y_all = load_data_simple(TRAIN_PATH)

    if len(X_all) == 0:
        print("HATA: Resimler bulunamadı!")
    else:
        # Veriyi Karıştır
        indices = np.arange(len(X_all))
        np.random.shuffle(indices)
        X_all = X_all[indices]
        y_all = y_all[indices]

        # %80 Eğitim, %20 Test
        split_idx = int(len(X_all) * 0.8)
        X_train, X_test = X_all[:split_idx], X_all[split_idx:]
        y_train, y_test = y_all[:split_idx], y_all[split_idx:]

        print(f"Model eğitiliyor ({len(X_train)} resim)...")
        my_model = create_model()
        my_model.fit(X_train, y_train, epochs=5, verbose=1)

        print("Panel Açılıyor!")
        root = tk.Tk()
        # Tüm verileri oyuna gönderiyoruz ki hafızayı kullanabilsin
        app = GameApp(root, my_model, X_train, y_train, X_test, y_test)
        root.mainloop()