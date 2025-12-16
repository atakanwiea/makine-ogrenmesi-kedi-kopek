# makine-ogrenmesi-kedi-kopek

Basit Cat vs Dog CNN projesi ve yardımcı scriptler.

## Hızlı başlangıç

1. Sanal ortam oluştur ve aktif et

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Bağımlılıkları yükle

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Script'i çalıştır

```powershell
python .\cat_dogs_ml.py
```

## Notlar
- `training_set/` ve `test_set/` gibi büyük veri klasörlerini repoya eklemeyin; `.gitignore` zaten hariç tutuyor.
- Eğer push sırasında yetkilendirme istenirse `gh auth login` ile GitHub CLI kullan veya bir Personal Access Token (PAT) oluşturup kullan.