import cv2
import numpy as np
import glob
import os
import json
 
# ============ AYARLAR (kendi checkerboard'unuza gore duzenleyin) ============
 
# Ic kose sayisi (kare sayisi degil!). Ornek: 10x7 kare -> 9x6 ic kose
CHECKERBOARD = (9, 6)  # (genislik, yukseklik) ic kose sayisi
 
# Bir karenin GERCEK kenar uzunlugu, milimetre cinsinden (cetvelle olcun!)
SQUARE_SIZE_MM = 21.0
 
# Kalibrasyon fotograflarinin bulundugu klasor
IMAGES_FOLDER = "calibration_images"
 
# Desteklenen dosya uzantilari
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]
 
# Sonuclarin kaydedilecegi dosya
OUTPUT_FILE = "kamera_parametreleri.json"
 
# =============================================================================
 
 
def kalibrasyonu_calistir():
    # Checkerboard'un 3D gercek dunya koordinatlarini hazirla (Z=0 duzleminde)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # gercek dunya birimine (mm) olcekle
 
    objpoints = []  # 3D gercek dunya noktalari (her gorsel icin)
    imgpoints = []  # 2D goruntu duzlemi noktalari (her gorsel icin)
 
    # Klasordeki tum gorselleri topla
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(IMAGES_FOLDER, ext)))
 
    if len(image_files) == 0:
        print(f"HATA: '{IMAGES_FOLDER}' klasorunde hic gorsel bulunamadi.")
        print("Once checkerboard fotograflarinizi bu klasore koyun.")
        return
 
    print(f"{len(image_files)} gorsel bulundu. Isleniyor...\n")
 
    gecerli_sayisi = 0
    img_shape = None
 
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print(f"  [ATLA] Okunamadi: {fname}")
            continue
 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (genislik, yukseklik)
 
        # Checkerboard koselerini bul
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
 
        if found:
            # Kose konumlarini alt-piksel hassasiyetinde iyilestir
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
 
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            gecerli_sayisi += 1
            print(f"  [OK]   Checkerboard bulundu: {os.path.basename(fname)}")
        else:
            print(f"  [YOK]  Checkerboard bulunamadi: {os.path.basename(fname)}")
 
    print(f"\nToplam {gecerli_sayisi}/{len(image_files)} gorselde checkerboard basariyla tespit edildi.")
 
    if gecerli_sayisi < 10:
        print("UYARI: 10'dan az gecerli gorsel var. Kalibrasyon hassasiyeti dusuk olabilir.")
        print("Daha fazla, farkli acilardan cekilmis fotograf eklemeniz onerilir.")
 
    if gecerli_sayisi < 3:
        print("HATA: Kalibrasyon icin yeterli gorsel yok (en az birkac tane gerekli). Durduruluyor.")
        return
 
    # Kalibrasyonu calistir
    print("\nKalibrasyon hesaplaniyor...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
 
    if not ret:
        print("HATA: Kalibrasyon basarisiz oldu.")
        return
 
    # Sonuclari ekrana yazdir
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
 
    print("\n" + "=" * 50)
    print("KALIBRASYON SONUCLARI")
    print("=" * 50)
    print(f"Goruntu boyutu (genislik x yukseklik): {img_shape}")
    print(f"f_x (piksel): {fx:.3f}")
    print(f"f_y (piksel): {fy:.3f}")
    print(f"c_x (piksel): {cx:.3f}")
    print(f"c_y (piksel): {cy:.3f}")
    print(f"Distorsiyon katsayilari: {dist_coeffs.ravel()}")
    print(f"Ortalama yeniden izdusum hatasi (px, dusuk = iyi): {ret:.4f}")
    print("=" * 50)
 
    # Sonuclari JSON dosyasina kaydet (sonraki adimlarda kullanmak icin)
    sonuc = {
        "image_width": int(img_shape[0]),
        "image_height": int(img_shape[1]),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "distortion_coefficients": dist_coeffs.ravel().tolist(),
        "reprojection_error_px": float(ret),
        "square_size_mm": SQUARE_SIZE_MM,
        "checkerboard_inner_corners": list(CHECKERBOARD),
        "num_images_used": gecerli_sayisi,
    }
 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
 
    print(f"\nParametreler '{OUTPUT_FILE}' dosyasina kaydedildi.")
    print("Bu dosyayi bir sonraki adimda (mesafe hesaplama) kullanacagiz.")
 
 
if __name__ == "__main__":
    kalibrasyonu_calistir()
 
