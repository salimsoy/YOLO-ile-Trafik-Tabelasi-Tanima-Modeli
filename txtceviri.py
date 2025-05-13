import pandas as pd
import os

# CSV'yi oku
df = pd.read_csv('dataset/Train.csv')

# Dönüştürme fonksiyonu
def convert_to_yolo(row):
    x_center = ((int(row['Roi.X1']) + int(row['Roi.X2'])) / 2) / int(row['Width'])
    y_center = ((int(row['Roi.Y1']) + int(row['Roi.Y2'])) / 2) / int(row['Height'])
    box_width = abs(int(row['Roi.X2']) - int(row['Roi.X1'])) / int(row['Width'])
    box_height = abs(int(row['Roi.Y2']) - int(row['Roi.Y1'])) / int(row['Height'])
    return int(row['ClassId']), x_center, y_center, box_width, box_height

# Klasör oluştur
os.makedirs("labels", exist_ok=True)

# Grupla ve .txt dosyalarına yaz
for path, group in df.groupby('Path'):
    filename = os.path.splitext(os.path.basename(path))[0] + ".txt"
    filepath = os.path.join("labels", filename)
    
    with open(filepath, 'w') as f:
        for _, row in group.iterrows():
            class_id, x, y, w, h = convert_to_yolo(row)
            if 0 <= class_id <= 10:
                continue
            class_id = class_id + 4
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # Eğer dosya boşsa sil
    if os.path.getsize(filepath) == 0:
        os.remove(filepath)