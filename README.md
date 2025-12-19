# Rust Person Detector (YOLOv8)

Bu proje, Rust oyunu için eğitilmiş YOLOv8s modelini kullanarak ekran üzerinde gerçek zamanlı insan (oyuncu) tespiti yapar.

## Kurulum (Sadece Tespit İçin)

Sadece `screen_detect.py` dosyasını çalıştırmak istiyorsanız, aşağıdakileri yapın:

1. **Gerekli Kütüphaneleri Yükleyin:**
   Aşağıdaki komutu terminalde çalıştırarak sadece tespit için gerekli paketleri yükleyebilirsiniz:
   
   ```bash
   pip install -r requirements_detect.txt
   ```

   Veya manuel olarak şu paketleri yükleyebilirsiniz:
   ```bash
   pip install ultralytics opencv-python numpy Pillow
   ```

## Kullanım

Ekran tespitini başlatmak için:

```bash
python screen_detect.py
```

### Özellikler (Screen Detect)
*   **Kalite Modları:** 640px ile 384px arasında seçim yapabilirsiniz.
*   **Güven Ayarı:** Tespit hassasiyetini (Confidence) ayarlayabilirsiniz.
*   **Kontroller:** Çalışırken `1-2-3-4` tuşları ile boyut değiştirebilir, `q` ile çıkabilirsiniz.

## Dosyalar
*   `screen_detect.py`: Canlı ekran tespiti yapan ana script.
*   `requirements_detect.txt`: Sadece tespit scripti için gerekli kütüphaneler.
*   `runs/detect/.../best.pt`: Eğitilmiş model dosyası.
