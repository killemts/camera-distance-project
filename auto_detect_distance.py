import cv2
import numpy as np
import json
import os
 
# ============ AYARLAR ============
 
CALIBRATION_FILE = "kamera_parametreleri.json"
 
REAL_WIDTH_CM = 47.0
REAL_HEIGHT_CM = 82.0
 
TEST_IMAGES = [
    {"path": "test_images_angled/foto100_1.jpeg", "real_distance_cm": 100},
    {"path": "test_images_angled/foto100_2.jpeg", "real_distance_cm": 100},
    {"path": "test_images_angled/foto100_3.jpeg", "real_distance_cm": 100},
    {"path": "test_images_angled/foto100_4.jpeg", "real_distance_cm": 100},
]
 
# Canny kenar tespiti esik degerleri (goruntu cok karanlik/aydinlikysa
# bu degerlerle oynamak gerekebilir)
CANNY_LOW = 50
CANNY_HIGH = 150
 
# Bulunan dortgen adaylarinin gercek en-boy oranina ne kadar yakin
# olmasi gerektigi (0.35 = %35 sapmaya kadar kabul et)
ASPECT_RATIO_TOLERANCE = 0.35
 
# ==========================================================================
 
 
def order_corners(pts):
    """4 noktayi Sol-Ust, Sag-Ust, Sag-Alt, Sol-Alt sirasina sokar."""
    pts = np.array(pts, dtype=np.float64)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
 
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
 
    return [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]
 
 
def refine_corners_subpixel(gray, corners):
    """Bulunan koseleri, komsu piksellerin gradyanina bakarak alt-piksel
    hassasiyetinde (daha keskin/gercek koseye dogru) iyilestirir."""
    corners_arr = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(gray, corners_arr, (11, 11), (-1, -1), criteria)
    return [tuple(pt[0]) for pt in refined]
 
 
def auto_detect_window(image, real_width_cm, real_height_cm, tolerance):
    """Kenar tespiti + kontur analizi ile pencereye benzer en iyi
    dortgeni bulmaya calisir. Bulunursa 4 kose (sirali), bulunamazsa
    None dondurur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)
 
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
 
    target_ratio = real_width_cm / real_height_cm
    img_area = image.shape[0] * image.shape[1]
 
    best_candidate = None
    best_score = float("inf")
 
    for cnt in contours[:30]:  # sadece en buyuk 30 konturu incele (hiz icin)
        area = cv2.contourArea(cnt)
        if area < img_area * 0.03:  # goruntunun %3'unden kucukse atla
            continue
 
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
 
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
 
        pts = approx.reshape(4, 2)
        ordered = order_corners(pts)
        tl, tr, br, bl = ordered
 
        w1 = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
        w2 = np.hypot(br[0] - bl[0], br[1] - bl[1])
        h1 = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
        h2 = np.hypot(br[0] - tr[0], br[1] - tr[1])
 
        avg_w = (w1 + w2) / 2
        avg_h = (h1 + h2) / 2
        if avg_h == 0:
            continue
 
        found_ratio = avg_w / avg_h
        ratio_diff = abs(found_ratio - target_ratio) / target_ratio
 
        if ratio_diff > tolerance:
            continue
 
        # skor: oran farki ne kadar kucukse o kadar iyi (ayni zamanda
        # buyuk alanlari hafif odullendiriyoruz)
        score = ratio_diff - (area / img_area) * 0.1
 
        if score < best_score:
            best_score = score
            best_candidate = ordered
 
    return best_candidate
 
 
corner_points = []
 
 
def mouse_callback(event, x, y, flags, param):
    global corner_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corner_points) < 4:
            corner_points.append((x, y))
            print(f"  Nokta {len(corner_points)}: ({x}, {y})")
 
 
def pick_corners_manual(image):
    global corner_points
    corner_points = []
    window_name = "ELLE SECIM: Sol-Ust, Sag-Ust, Sag-Alt, Sol-Alt (q: erken bitir)"
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
    return corner_points.copy() if len(corner_points) == 4 else None
 
 
def confirm_detection(image, corners):
    """Bulunan dortgeni ekranda gosterip kullanicidan onay ister."""
    display = image.copy()
    pts = np.array(corners, np.int32)
    cv2.polylines(display, [pts], True, (0, 255, 0), 4)
    for i, pt in enumerate(corners):
        cv2.circle(display, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
 
    window_name = "Bu dogru mu? Y = Evet, R = Hayir (elle sec)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 1000)
    cv2.imshow(window_name, display)
 
    print("  Tespit edilen dortgen ekranda gosteriliyor.")
    print("  Dogruysa 'y', yanlissa 'r' tuslayip Enter'a basmadan sadece harfe basin.")
 
    result = None
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            result = True
            break
        elif key == ord('r'):
            result = False
            break
 
    cv2.destroyAllWindows()
    return result
 
 
def get_scaled_camera_matrix(calib, img_w, img_h):
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
        return None, None
 
    distance_z = float(tvec[2][0])
    distance_euclid = float(np.linalg.norm(tvec))
    return distance_z, distance_euclid
 
 
def main():
    if not os.path.exists(CALIBRATION_FILE):
        print(f"HATA: '{CALIBRATION_FILE}' bulunamadi.")
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
        print(f"\n=== {path} ({img_w}x{img_h}) ===")
 
        print("  Otomatik tespit deneniyor...")
        corners = auto_detect_window(image, REAL_WIDTH_CM, REAL_HEIGHT_CM, ASPECT_RATIO_TOLERANCE)
 
        if corners is not None:
            gray_for_refine = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            try:
                corners = refine_corners_subpixel(gray_for_refine, corners)
                print("  Koseler alt-piksel hassasiyetinde iyilestirildi.")
            except cv2.error:
                print("  [UYARI] Alt-piksel iyilestirme basarisiz oldu, ham koseler kullaniliyor.")
 
            confirmed = confirm_detection(image, corners)
            if not confirmed:
                print("  Otomatik tespit reddedildi, elle secime geciliyor...")
                corners = pick_corners_manual(image)
        else:
            print("  Otomatik tespit basarisiz oldu, elle secime geciliyor...")
            corners = pick_corners_manual(image)
 
        if corners is None or len(corners) < 4:
            print("  [ATLA] Kose bulunamadi.")
            continue
 
        camera_matrix, dist_coeffs = get_scaled_camera_matrix(calib, img_w, img_h)
        dist_z, dist_euclid = solve_distance(
            corners, REAL_WIDTH_CM, REAL_HEIGHT_CM, camera_matrix, dist_coeffs
        )
 
        if dist_z is None:
            print("  [HATA] solvePnP cozemedi.")
            continue
 
        print(f"  Z-ekseni (derinlik) mesafesi: {dist_z:.1f} cm")
        print(f"  Oklid mesafesi: {dist_euclid:.1f} cm")
 
        if real_dist:
            error_z = abs(dist_z - real_dist)
            print(f"  Gercek mesafe: {real_dist} cm")
            print(f"  HATA (Z-ekseni): {error_z:.1f} cm ({error_z/real_dist*100:.1f}%)")
 
    print("\nTum test gorselleri islendi.")
 
 
if __name__ == "__main__":
    main()
 