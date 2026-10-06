
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
 
# ============ AYARLAR ============
 
CALIBRATION_FILE = "kamera_parametreleri.json"
MODEL_FILE = "best.pt"
 
# Modelin camicin kullandigi sinif numarasi (model.names ile dogruladik)
GLASS_CLASS_ID = 0
 
# Bu esigin altindaki tespitler goz ardi edilir (0.0 - 1.0 arasi)
CONFIDENCE_THRESHOLD = 0.4
 
REAL_WIDTH_CM = 47.0
REAL_HEIGHT_CM = 82.0
 
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
 
 
def pick_corners_manual(image):
    """YOLO cami bulamazsa devreye giren yedek plan: elle 4 kose secimi."""
    global corner_points
    corner_points = []
    window_name = "YOLO BULAMADI - ELLE SEC: Sol-Ust, Sag-Ust, Sag-Alt, Sol-Alt"
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
 
 
def detect_glass_yolo(model, image, real_width_cm, real_height_cm, ratio_tolerance=0.3, debug=False):
    """YOLO ile camin kutusunu bulur. Birden fazla benzer pencere/cam
    varsa, sadece en yuksek guven skoruna degil, ayni zamanda camin
    bilinen gercek en-boy oranina (real_width_cm/real_height_cm) en
    yakin olan tespite de bakarak dogru pencereyi secmeye calisir."""
    results = model(image, verbose=False, conf=0.05)
    target_ratio = real_width_cm / real_height_cm
 
    if debug:
        print("  --- HAM TESPITLER (esik uygulanmadan) ---")
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names.get(cls_id, str(cls_id))
                xyxy = box.xyxy[0].cpu().numpy()
                print(f"    sinif={cls_name} (id={cls_id})  guven={conf:.3f}  "
                      f"kutu=({xyxy[0]:.0f},{xyxy[1]:.0f})-({xyxy[2]:.0f},{xyxy[3]:.0f})")
        print("  ------------------------------------------")
 
    candidates = []
 
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
 
            if cls_id != GLASS_CLASS_ID:
                continue
            if conf < CONFIDENCE_THRESHOLD:
                continue
 
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            w_px = x2 - x1
            h_px = y2 - y1
            if h_px <= 0:
                continue
 
            found_ratio = w_px / h_px
            ratio_diff = abs(found_ratio - target_ratio) / target_ratio
 
            candidates.append({
                "box": (x1, y1, x2, y2),
                "conf": conf,
                "ratio_diff": ratio_diff,
            })
 
    if not candidates:
        return None, 0.0, []
 
    in_tolerance = [c for c in candidates if c["ratio_diff"] <= ratio_tolerance]
 
    if in_tolerance:
        # Guven skoruna degil, GERCEK ORANA EN YAKIN olana oncelik ver --
        # bu, "hangi cam bizim aradigimiz" sorusunu guven skorundan daha
        # guvenilir sekilde cevapliyor (guven skoru sadece "bu bir cam mi"
        # sorusuna cevap veriyor, "hangi cam" sorusuna degil).
        best = min(in_tolerance, key=lambda c: c["ratio_diff"])
    else:
        best = min(candidates, key=lambda c: c["ratio_diff"])
        print(f"  [UYARI] Hicbir tespit beklenen orana ({target_ratio:.2f}) tam uymuyor, "
              f"en yakin secildi (oran farki: {best['ratio_diff']*100:.0f}%).")
 
    return best["box"], best["conf"], candidates
 
 
def show_detection(image, selected_box, selected_conf, all_candidates):
    """Tum adaylari ekranda gosterir: secilen kutu YESIL, diger elenen
    adaylar TURUNCU renkte cizilir -- boylece modelin gordugu her seyi
    ve neden bu kutunun secildigini gorsel olarak takip edebilirsiniz."""
    display = image.copy()
 
    # Once elenen (secilmeyen) adaylari turuncu ciz
    for cand in all_candidates:
        if cand["box"] == selected_box:
            continue
        x1, y1, x2, y2 = [int(v) for v in cand["box"]]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 3)
        label = f"elendi ({cand['conf']:.2f})"
        cv2.putText(display, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
 
    # En son, secilen kutuyu YESIL ve kalin ciz (ustte kalsin diye en son)
    x1, y1, x2, y2 = [int(v) for v in selected_box]
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 4)
    label = f"SECILEN: balustrade_glass {selected_conf:.2f}"
    cv2.putText(display, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
 
    window_name = "Yesil = secilen, Turuncu = elenen - devam icin bir tusa basin"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 1000)
    cv2.imshow(window_name, display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
def get_scaled_focal_lengths(calib, img_w, img_h):
    calib_w, calib_h = calib["image_width"], calib["image_height"]
    fx, fy = calib["fx"], calib["fy"]
 
    scale_x = img_w / calib_w
    scale_y = img_h / calib_h
 
    return fx * scale_x, fy * scale_y
 
 
def compute_distance_from_box(box, fx, fy, real_width_cm, real_height_cm):
    x1, y1, x2, y2 = box
    width_px = x2 - x1
    height_px = y2 - y1
 
    dist_from_width = (real_width_cm * fx) / width_px
    dist_from_height = (real_height_cm * fy) / height_px
 
    return dist_from_width, dist_from_height, width_px, height_px
 
 
def main():
    if not os.path.exists(CALIBRATION_FILE):
        print(f"HATA: '{CALIBRATION_FILE}' bulunamadi.")
        return
    if not os.path.exists(MODEL_FILE):
        print(f"HATA: '{MODEL_FILE}' bulunamadi. best.pt dosyasini proje klasorune koyun.")
        return
 
    with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
        calib = json.load(f)
 
    print("YOLO modeli yukleniyor...")
    model = YOLO(MODEL_FILE)
    print(f"Model sinifla ri: {model.names}\n")
 
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
 
        print("  YOLO ile tespit yapiliyor...")
        box, conf, all_candidates = detect_glass_yolo(model, image, REAL_WIDTH_CM, REAL_HEIGHT_CM, debug=True)
 
        if box is None:
            print(f"  [UYARI] YOLO 'balustrade_glass' bulamadi (esik: {CONFIDENCE_THRESHOLD}).")
            print("  Elle secime geciliyor...")
            corners = pick_corners_manual(image)
            if corners is None:
                print("  [ATLA] Kose secilmedi.")
                continue
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            box = (min(xs), min(ys), max(xs), max(ys))
        else:
            print(f"  YOLO tespiti: guven skoru = {conf:.2f}")
            print(f"  Kutu: ({box[0]:.0f}, {box[1]:.0f}) - ({box[2]:.0f}, {box[3]:.0f})")
            show_detection(image, box, conf, all_candidates)
 
        fx, fy = get_scaled_focal_lengths(calib, img_w, img_h)
        dist_w, dist_h, w_px, h_px = compute_distance_from_box(
            box, fx, fy, REAL_WIDTH_CM, REAL_HEIGHT_CM
        )
        dist_avg = (dist_w + dist_h) / 2
 
        print(f"  Piksel genislik: {w_px:.1f} px | Piksel yukseklik: {h_px:.1f} px")
        print(f"  Genislikten hesaplanan mesafe: {dist_w:.1f} cm")
        print(f"  Yukseklikten hesaplanan mesafe: {dist_h:.1f} cm")
        print(f"  ORTALAMA TAHMINI MESAFE: {dist_avg:.1f} cm")
 
        if real_dist:
            error = abs(dist_avg - real_dist)
            error_pct = (error / real_dist) * 100
            print(f"  Gercek mesafe: {real_dist} cm")
            print(f"  HATA: {error:.1f} cm ({error_pct:.1f}%)")
 
    print("\nTum test gorselleri islendi.")
 
 
if __name__ == "__main__":
    main()
 