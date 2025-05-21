import cv2
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO("run/detect/train/weights/best1.pt")  # Yolunu kendi modeline göre güncelle

# Webcam'i başlat (0 = varsayılan kamera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO modeline input olarak ver (BGR görüntü kullanılır)
    results = model.predict(source=frame, show=False, conf=0.5)

    # Tahmin edilen kutucukları çiz
    annotated_frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow("YOLOv8 - Webcam Detection", annotated_frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
