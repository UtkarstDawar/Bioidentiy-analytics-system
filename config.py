# System Configuration Constants

# 1. Face Analysis Settings
# 'opencv' is fastest. Use 'ssd' or 'mtcnn' for higher accuracy (slower)
DETECTOR_BACKEND = 'opencv' 
MODEL_NAME = 'VGG-Face'

# 2. Dress Color Analysis Settings
# Defines the area relative to the face to scan for clothes
ROI_OFFSET_Y = 0.15   # Start 15% below the chin
ROI_HEIGHT = 0.6      # Scan 60% of the face height downwards
ROI_WIDTH = 1.4       # Scan 140% of the face width
ANALYSIS_RES = (64, 64) # Downsample ROI to 64px for 10x speed

# 3. UI Settings
PAGE_TITLE = "Bio-Identity Analytics"
THEME_COLOR = "#4facfe"