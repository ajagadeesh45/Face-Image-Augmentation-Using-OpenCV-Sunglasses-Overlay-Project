# Face Image Augmentation Using OpenCV – Sunglasses Overlay Project

# Aim

To overlay a sunglasses image onto a face image using OpenCV by applying image masking and transparency handling techniques for realistic image augmentation.

---

# Algorithm

### Step 1

Import the required libraries such as OpenCV, NumPy, and Matplotlib.

### Step 2

Load the face image where the sunglasses will be placed.

### Step 3

Load the sunglasses PNG image with the alpha channel to preserve transparency.

### Step 4

Resize the sunglasses image according to the eye region size of the face image.

### Step 5

Separate the BGR channels and alpha channel from the sunglasses image.

### Step 6

Apply the naive direct overlay method by replacing the eye region with the sunglasses image.

### Step 7

Create a mask using the alpha channel of the PNG image.

### Step 8

Extract the eye Region of Interest (ROI) from the face image.

### Step 9

Remove the background from the ROI using the mask.

### Step 10

Keep only the sunglass foreground using the mask.

### Step 11

Combine the ROI and sunglass foreground using arithmetic operations.

### Step 12

Replace the final blended ROI back into the original face image.

### Step 13

Display the final output image with realistic sunglasses overlay.

---

# Program

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load face image
faceImage = cv2.imread("myphoto.jpg")

# Convert BGR to RGB for display
faceImageRGB = faceImage[:, :, ::-1]

# Display original image
plt.figure(figsize=[8,8])
plt.imshow(faceImageRGB)
plt.title("Original Face Image")
plt.axis("off")
plt.show()

# Load sunglasses PNG with alpha channel
glassPNG = cv2.imread("glass.png", -1)

# Resize sunglasses
glassPNG = cv2.resize(glassPNG, (100, 35))

# Separate BGR and alpha channels
glassBGR = glassPNG[:, :, 0:3]
glassMask1 = glassPNG[:, :, 3]

# -------------------------------
# Method 1: Naive Overlay Method
# -------------------------------

faceWithGlassesNaive = faceImage.copy()

# Replace eye region directly
faceWithGlassesNaive[100:135, 90:190] = glassBGR

plt.figure(figsize=[8,8])
plt.imshow(faceWithGlassesNaive[:, :, ::-1])
plt.title("Naive Overlay Output")
plt.axis("off")
plt.show()

# ----------------------------------------
# Method 2: Arithmetic + Masking Method
# ----------------------------------------

faceWithGlassesArithmetic = faceImage.copy()

# Create 3-channel mask
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Normalize mask
glassMask = np.uint8(glassMask / 255)

# Extract ROI
eyeROI = faceWithGlassesArithmetic[100:135, 90:190]

# Remove background
maskedEye = cv2.multiply(eyeROI, (1 - glassMask))

# Keep only sunglass foreground
maskedGlass = cv2.multiply(glassBGR, glassMask)

# Add both images
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Replace ROI
faceWithGlassesArithmetic[100:135, 90:190] = eyeRoiFinal

# Display final output
plt.figure(figsize=[8,8])
plt.imshow(faceWithGlassesArithmetic[:, :, ::-1])
plt.title("Final Output Using Masking Method")
plt.axis("off")
plt.show()
