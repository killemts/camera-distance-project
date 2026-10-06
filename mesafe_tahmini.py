import cv2
import numpy as np
import json
import os
 
# ============ AYARLAR (kendi olcumlerinize gore duzenleyin) ============
 
CALIBRATION_FILE = "kamera_parametreleri.json"
 
# Camin (cercevesiz) gercek boyutlari, santimetre cinsinden
REAL_WIDTH_CM = 47.0
REAL_HEIGHT_CM = 82.0
 
# Test edilecek goruntuler. "real_distance_cm" biliniyorsa dogrulama icin
# yazin (elle olctugunuz gercek mesafe); bilmiyorsaniz None birakin.
TEST_IMAGES = [
    {"path": "test_images/foto75.jpeg", "real_distance_cm": 75},
    {"path": "test_images/foto100.jpeg", "real_distance_cm": 100},
    {"path": "test_images/foto125.jpeg", "real_distance_cm": 125},
]
 
# ==========================================================================
 
 
corner_points = []
 
 
def mouse_callback(event, x, y, flags, param):
    global corner_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corner_points) < 4:
            corner_points.append((x, y))
            print(f"  Nokta {len(corner_points)}: ({x}, {y})")
 
 
def pick_corners(image):
    global corner_points
    corner_points = []
    window_name = "Sirayla tikla: Sol-Ust, Sag-Ust, Sag-Alt, Sol-Alt  (q: erken bitir)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 1000)
    cv2.setMouseCallback(window_name, mouse_callback)
 
    while True:
        display = image.copy()
        for i, pt in enumerate(corner_points):
            cv2.circle(display, pt, 10, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 12, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if len(corner_points) == 4:
            pts = np.array(corner_points, np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 3)
 
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or len(corner_points) == 4:
            break
 
    cv2.destroyAllWindows()
    return corner_points.copy()
 
 
def get_scaled_camera_params(calib, img_w, img_h):
    """Kalibrasyon fotografiyla test fotografinin cozunurlugu/yonelimi
    farkliysa fx, fy, cx, cy degerlerini uygun sekilde olcekler."""
    calib_w, calib_h = calib["image_width"], calib["image_height"]
    calib_landscape = calib_w > calib_h
    img_landscape = img_w > img_h
 
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
 
    if calib_landscape == img_landscape:
        scale_x = img_w / calib_w
        scale_y = img_h / calib_h
        fx2, fy2 = fx * scale_x, fy * scale_y
        cx2, cy2 = cx * scale_x, cy * scale_y
    else:
        # Kalibrasyon ve test fotografi farkli yonelimde (biri yatay,
        # digeri dikey) -- 90 derece donuk kabul edip donduruyoruz.
        eff_calib_w, eff_calib_h = calib_h, calib_w
        scale_x = img_w / eff_calib_w
        scale_y = img_h / eff_calib_h
        fx2, fy2 = fy * scale_x, fx * scale_y
        cx2, cy2 = cy * scale_x, cx * scale_y
        print("  [UYARI] Foto ile kalibrasyon farkli yonelimde (dikey/yatay). "
              "Odak degerleri tahmini donduruldu; ideal degil ama kullanilabilir.")
 
    return fx2, fy2, cx2, cy2
 
 
def compute_distance(corners, fx, fy, real_width_cm, real_height_cm):
    (tl, tr, br, bl) = corners
 
    width_top = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    width_bottom = np.hypot(br[0] - bl[0], br[1] - bl[1])
    height_left = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
    height_right = np.hypot(br[0] - tr[0], br[1] - tr[1])
 
    avg_width_px = (width_top + width_bottom) / 2
    avg_height_px = (height_left + height_right) / 2
 
    dist_from_width_cm = (real_width_cm * fx) / avg_width_px
    dist_from_height_cm = (real_height_cm * fy) / avg_height_px
 
    return dist_from_width_cm, dist_from_height_cm, avg_width_px, avg_height_px
 
 
def main():
    if not os.path.exists(CALIBRATION_FILE):
        print(f"HATA: '{CALIBRATION_FILE}' bulunamadi. Once kalibrasyon scriptini calistirin.")
        return
 
    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        calib = json.load(f)
 
    for item in TEST_IMAGES:
        path = item["path"]
        real_dist = item.get("real_distance_cm")
 
        if not os.path.exists(path):
            print(f"\n[ATLA] Dosya bulunamadi: {path}")
            continue
 
        image = cv2.imread(path)
        if image is None:
            print(f"\n[ATLA] Okunamadi: {path}")
            continue
 
        img_h, img_w = image.shape[:2]
        fx, fy, cx, cy = get_scaled_camera_params(calib, img_w, img_h)
 
        print(f"\n=== {path} ({img_w}x{img_h}) ===")
        print("Camin 4 kosesini SIRAYLA tikla: Sol-Ust -> Sag-Ust -> Sag-Alt -> Sol-Alt")
        corners = pick_corners(image)
 
        if len(corners) < 4:
            print("  [ATLA] 4 kose secilmedi.")
            continue
 
        dist_w, dist_h, w_px, h_px = compute_distance(
            corners, fx, fy, REAL_WIDTH_CM, REAL_HEIGHT_CM
        )
        dist_avg = (dist_w + dist_h) / 2
 
        print(f"  Piksel genislik: {w_px:.1f} px | Piksel yukseklik: {h_px:.1f} px")
        print(f"  Genislikten hesaplanan mesafe: {dist_w:.1f} cm")
        print(f"  Yukseklikten hesaplanan mesafe: {dist_h:.1f} cm")
        print(f"  ORTALAMA TAHMINI MESAFE: {dist_avg:.1f} cm")
 
        if real_dist:
            error = abs(dist_avg - real_dist)
            error_pct = (error / real_dist) * 100
            print(f"  Gercek (elle olculen) mesafe: {real_dist} cm")
            print(f"  HATA: {error:.1f} cm ({error_pct:.1f}%)")
 
    print("\nTum test gorselleri islendi.")
 
 
if __name__ == "__main__":
    main()
 