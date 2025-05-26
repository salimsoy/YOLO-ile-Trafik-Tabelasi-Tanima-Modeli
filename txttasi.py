import os
import shutil

# Ana klasör yolu (içinde .jpg ve .txt dosyaları var)
main_folder = "C:/Users/salim/Desktop/python-kodlari/trafik_algila/ts"

# 'txt' adında alt klasör oluştur
txt_folder = os.path.join(main_folder, "txt")
os.makedirs(txt_folder, exist_ok=True)

# Dosyaları kontrol et
for filename in os.listdir(main_folder):
    if filename.endswith(".txt"):
        src_path = os.path.join(main_folder, filename)
        dst_path = os.path.join(txt_folder, filename)

        # Dosyayı taşı
        shutil.move(src_path, dst_path)
        print(f"{filename} → 'txt' klasörüne taşındı.")

print("Tüm .txt dosyaları taşındı.")
