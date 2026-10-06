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
    {"path": "test_images_angled/foto100_1.jpeg", "real_distance_cm": 100},
    {"path": "test_images_angled/foto100_2.jpeg", "real_distance_cm": 100},
    {"path": "test_images_angled/foto100_3.jpeg", "real_distance_cm": 100},
    {"path": "test_images_angled/foto100_4.jpeg", "real_distance_cm": 100},
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
 
 
def get_scaled_camera_matrix(calib, img_w, img_h):
    """Kalibrasyon ile test fotografinin cozunurlugu farkliysa, kamera
    matrisini orantili olarak olcekler (ayni en-boy oraniysa dogru calisir)."""
    calib_w, calib_h = calib["image_width"], calib["image_height"]
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
 
    scale_x = img_w / calib_w
    scale_y = img_h / calib_h
 
    camera_matrix = np.array([
        [fx * scale_x, 0, cx * scale_x],
        [0, fy * scale_y, cy * scale_y],
        [0, 0, 1]
    ], dtype=np.float64)
 
    dist_coeffs = np.array(calib["distortion_coefficients"], dtype=np.float64)
 
    return camera_matrix, dist_coeffs
 
 
def solve_distance(corners_2d, real_width_cm, real_height_cm, camera_matrix, dist_coeffs):
    # Camin gercek dunya 3D koordinatlari (dikdortgen bir duzlem, Z=0).
    # Orijin (0,0,0) camin TAM ORTASINA denk gelecek sekilde tanimlandi
    # (kose yerine merkez), boylece Oklid mesafesi de anlamli olur.
    # Sira: Sol-Ust, Sag-Ust, Sag-Alt, Sol-Alt
    half_w = real_width_cm / 2.0
    half_h = real_height_cm / 2.0
    object_points = np.array([
        [-half_w, -half_h, 0],
        [half_w, -half_h, 0],
        [half_w, half_h, 0],
        [-half_w, half_h, 0],
    ], dtype=np.float64)
 
    image_points = np.array(corners_2d, dtype=np.float64)
 
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE
    )
 
    if not success:
        return None, None, None
 
    # tvec: kameranin cam duzlemine gore konumu (cm cinsinden, X,Y,Z)
    # Dogrudan mesafe = kameradan cam merkezine olan Oklid mesafesi degil,
    # tvec zaten kamera koordinat sisteminde camin konumunu verir; Z bileseni
    # kabaca "derinlik/mesafe", ama acili bakislarda daha dogrusu vektor normu.
    distance_z_only = float(tvec[2][0])
    distance_euclidean = float(np.linalg.norm(tvec))
 
    return distance_z_only, distance_euclidean, tvec
 
 
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
        camera_matrix, dist_coeffs = get_scaled_camera_matrix(calib, img_w, img_h)
 
        print(f"\n=== {path} ({img_w}x{img_h}) ===")
        print("Camin 4 kosesini SIRAYLA tikla: Sol-Ust -> Sag-Ust -> Sag-Alt -> Sol-Alt")
        corners = pick_corners(image)
 
        if len(corners) < 4:
            print("  [ATLA] 4 kose secilmedi.")
            continue
 
        dist_z, dist_euclid, tvec = solve_distance(
            corners, REAL_WIDTH_CM, REAL_HEIGHT_CM, camera_matrix, dist_coeffs
        )
 
        if dist_z is None:
            print("  [HATA] solvePnP cozemedi.")
            continue
 
        print(f"  Z-ekseni (derinlik) mesafesi: {dist_z:.1f} cm")
        print(f"  Oklid (dogrudan) mesafesi: {dist_euclid:.1f} cm")
 
        if real_dist:
            error_z = abs(dist_z - real_dist)
            error_euclid = abs(dist_euclid - real_dist)
            print(f"  Gercek mesafe: {real_dist} cm")
            print(f"  HATA (Z-ekseni): {error_z:.1f} cm ({error_z/real_dist*100:.1f}%)")
            print(f"  HATA (Oklid): {error_euclid:.1f} cm ({error_euclid/real_dist*100:.1f}%)")
 
    print("\nTum test gorselleri islendi.")
 
 
if __name__ == "__main__":
    main()