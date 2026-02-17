import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import config

class BioAnalyticEngine:
    """Core AI wrapper for DeepFace."""
    
    @staticmethod
    def scan_subject(img_rgb):
        try:
            # Run deep learning inference
            # enforce_detection=True ensures we only return data if a face exists
            results = DeepFace.analyze(
                img_path=np.array(img_rgb), 
                actions=['age', 'race', 'emotion'], 
                detector_backend=config.DETECTOR_BACKEND,
                enforce_detection=True,
                silent=True
            )
            
            # DeepFace returns a list of faces; we take the first one
            return results[0] if isinstance(results, list) else results
            
        except Exception:
            return None

class ChromaticEngine:
    """Handles color extraction logic without external dependencies."""
    
    def __init__(self):
        # Internal database of standard colors to avoid library errors
        self.palette = {
            "Black": (0, 0, 0), "White": (255, 255, 255), "Red": (255, 0, 0),
            "Lime": (0, 255, 0), "Blue": (0, 0, 255), "Yellow": (255, 255, 0),
            "Cyan": (0, 255, 255), "Magenta": (255, 0, 255), "Silver": (192, 192, 192),
            "Gray": (128, 128, 128), "Maroon": (128, 0, 0), "Olive": (128, 128, 0),
            "Green": (0, 128, 0), "Purple": (128, 0, 128), "Teal": (0, 128, 128),
            "Navy": (0, 0, 128), "Orange": (255, 165, 0), "Gold": (255, 215, 0),
            "Pink": (255, 192, 203), "Brown": (165, 42, 42), "Beige": (245, 245, 220)
        }
        
        # Prepare Spatial KD-Tree for O(log n) lookup speed
        self.color_names = list(self.palette.keys())
        self.rgb_values = list(self.palette.values())
        self.kdtree = KDTree(self.rgb_values)

    def _get_nearest_color(self, rgb_tuple):
        """Finds the mathematically closest color in the palette."""
        distance, index = self.kdtree.query(rgb_tuple)
        return self.color_names[index]

    def analyze_attire(self, image_bgr, face_meta):
        """
        Determines the dominant color of the clothing.
        """
        # Unpack face coordinates
        x, y, w, h = face_meta['x'], face_meta['y'], face_meta['w'], face_meta['h']
        H_img, W_img, _ = image_bgr.shape

        # Dynamic ROI Calculation (Chest Area)
        # We calculate the chest area relative to face size
        roi_start_y = int(y + h + (h * config.ROI_OFFSET_Y))
        roi_end_y = int(roi_start_y + (h * config.ROI_HEIGHT))
        
        center_x = x + w // 2
        roi_w_half = int((w * config.ROI_WIDTH) / 2)
        roi_start_x = max(0, center_x - roi_w_half)
        roi_end_x = min(W_img, center_x + roi_w_half)

        # Boundary Checks
        if roi_start_y >= H_img or roi_start_x >= roi_end_x:
            return "Not Visible", "#000000"

        # Crop the chest region
        roi = image_bgr[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
        
        if roi.size == 0:
             return "Not Visible", "#000000"

        # Optimization: Resize to small resolution for faster processing
        roi_small = cv2.resize(roi, config.ANALYSIS_RES, interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi_small, cv2.COLOR_BGR2RGB)
        pixel_list = roi_rgb.reshape((-1, 3))

        # K-Means Clustering to find dominant color
        kmeans = KMeans(n_clusters=1, n_init=5)
        kmeans.fit(pixel_list)
        
        dominant_rgb = kmeans.cluster_centers_[0].astype(int)
        
        # Match to human readable name
        color_name = self._get_nearest_color(dominant_rgb)
        
        # Return Name and Hex (for UI display)
        r, g, b = dominant_rgb
        hex_code = f"#{r:02x}{g:02x}{b:02x}"
        
        return color_name, hex_code