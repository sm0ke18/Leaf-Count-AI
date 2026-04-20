import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_PATH = 'leafcountbest.pt'  # Make sure best.pt is in the same folder as this script
IMAGE_PATH = 'G:\\LEAF AI\\testing_images\\5leaves.jpg' # Put the name of your image here

def analyze_leaves(img_path, model_path):
    # 1. Load Model
    model = YOLO(model_path)
    
    # 2. Run Inference
    results = model(img_path, conf=0.20, iou=0.7)[0]
    img_orig = cv2.imread(img_path)
    
    if results.masks is None:
        print("No leaves detected.")
        return

    # 3. Process Results
    annotated_frame = results.plot(labels=True, boxes=True)
    color_tally = {'Green': 0, 'Yellow': 0, 'Brown': 0}
    
    for mask_pts in results.masks.xy:
        # Create mask
        blank_mask = np.zeros(img_orig.shape[:2], dtype=np.uint8)
        cv2.fillPoly(blank_mask, np.int32([mask_pts]), 255)
        
        # Color Analysis
        leaf_pixels = cv2.bitwise_and(img_orig, img_orig, mask=blank_mask)
        hsv_leaf = cv2.cvtColor(leaf_pixels, cv2.COLOR_BGR2HSV)
        
        green = cv2.countNonZero(cv2.inRange(hsv_leaf, np.array([35, 40, 40]), np.array([85, 255, 255])))
        yellow = cv2.countNonZero(cv2.inRange(hsv_leaf, np.array([20, 40, 40]), np.array([34, 255, 255])))
        brown = cv2.countNonZero(cv2.inRange(hsv_leaf, np.array([10, 50, 20]), np.array([19, 255, 150])))
        
        # Tally dominant color
        counts = {'Green': green, 'Yellow': yellow, 'Brown': brown}
        dominant = max(counts, key=counts.get)
        color_tally[dominant] += 1

    # 4. Show Report
    print(f"\n--- Analysis for {img_path} ---")
    print(f"Total Leaves: {len(results.masks)}")
    for color, count in color_tally.items():
        print(f"{color}: {count}")

    # 5. Display Result
    cv2.imshow("Leaf Analysis Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_leaves(IMAGE_PATH, MODEL_PATH)