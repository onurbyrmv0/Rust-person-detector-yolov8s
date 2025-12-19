"""
Rust Oyunu Ekran Tespiti - ULTRA KALITE
Yüksek FPS + Yüksek Kalite + İyi Tespit
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
from PIL import ImageGrab
import ctypes

# DPI awareness için (yüksek çözünürlüklü ekranlarda doğru yakalama)
ctypes.windll.user32.SetProcessDPIAware()

def screen_capture_detect():
    """
    Ekrandan canlı karakter tespiti yapar - YÜKSEK KALİTE
    """
    
    # Modeli yükle
    print("🔄 Model yükleniyor...")
    model = YOLO('runs/detect/rust_person_detector5/weights/best.pt')
    model.to('cuda')
    print("✅ Model yüklendi (CUDA)!")
    
    # Ekran boyutunu al
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)
    
    print(f"\n📺 Ekran: {screen_width}x{screen_height}")
    
    # === FPS AYARLARI ===
    print("\n⚡ Kalite Modu Seçin:")
    print("   1: Maksimum Kalite (imgsz=640) - En iyi tespit")
    print("   2: Yüksek Kalite (imgsz=544)")
    print("   3: Dengeli (imgsz=480) [ÖNERİLEN]")
    print("   4: Hızlı (imgsz=384)")
    
    fps_choice = input("\nMod seçin (varsayılan 1): ").strip()
    
    imgsz_options = {'1': 640, '2': 544, '3': 480, '4': 384}
    imgsz = imgsz_options.get(fps_choice, 640)
    
    # Confidence ayarı
    print("\n🎯 Tespit Hassasiyeti:")
    print("   1: Düşük (conf=0.25) - Daha fazla tespit, bazı yanlışlar olabilir")
    print("   2: Orta (conf=0.35) [ÖNERİLEN]")
    print("   3: Yüksek (conf=0.5) - Sadece emin olunanlar")
    
    conf_choice = input("\nHassasiyet seçin (varsayılan 2): ").strip()
    conf_options = {'1': 0.25, '2': 0.35, '3': 0.5}
    conf_threshold = conf_options.get(conf_choice, 0.35)
    
    print(f"\n🎮 Ekran tespiti başlıyor...")
    print(f"   Ekran: {screen_width}x{screen_height}")
    print(f"   Model boyutu: {imgsz}px")
    print(f"   Confidence: {conf_threshold}")
    print(f"   Çalışırken 1-2-3-4 tuşlarıyla boyut değiştir")
    print(f"   Çıkmak için 'q' tuşuna basın")
    print("="*50)
    
    # Model ısınma
    print("🔥 Model ısınıyor...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(source=dummy, verbose=False, imgsz=imgsz, half=True)
    print("✅ Hazır!\n")
    
    # FPS hesaplama
    frame_times = []
    fps = 0
    
    # Pencere oluştur
    cv2.namedWindow("Rust Karakter Tespiti", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rust Karakter Tespiti", 960, 540)
    
    while True:
        t_start = time.perf_counter()
        
        # Ekran görüntüsü al (PIL - yüksek kalite)
        screenshot = ImageGrab.grab()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Tespit yap (OPTİMİZE)
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False,
            imgsz=imgsz,
            half=True,
            max_det=20,
            agnostic_nms=True,
            device=0
        )
        
        # Sonuçları çiz
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_conf = float(box.conf[0])
            # Kalın yeşil kutu
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Confidence label
            label = f'PERSON {box_conf:.0%}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-25), (x1+w+10, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1+5, y1-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # FPS hesapla (son 30 frame ortalaması)
        frame_times.append(time.perf_counter() - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = len(frame_times) / sum(frame_times)
        
        # Tespit sayısı
        num_det = len(results[0].boxes)
        
        # FPS rengi
        color = (0, 255, 0) if fps >= 30 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
        
        # Bilgi ekle
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Tespit: {num_det} karakter", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"[{imgsz}px] Conf:{conf_threshold}", (10, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Göster
        cv2.imshow("Rust Karakter Tespiti", frame)
        
        # Tuş kontrolü
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            imgsz = 640
        elif key == ord('2'):
            imgsz = 544
        elif key == ord('3'):
            imgsz = 480
        elif key == ord('4'):
            imgsz = 384
    
    cv2.destroyAllWindows()
    print(f"\n✅ Ortalama FPS: {fps:.1f}")


def main():
    print("\n" + "="*50)
    print("🎮 RUST Ekran Tespit - HIZLI VERSIYON")
    print("="*50)
    print("\n⚠️ Dikkat: Bu araç sadece eğitim amaçlıdır!")
    print("   Oyunlarda hile kullanmak yasaktır.\n")
    
    input("Başlamak için Enter'a basın...")
    screen_capture_detect()


if __name__ == '__main__':
    main()
