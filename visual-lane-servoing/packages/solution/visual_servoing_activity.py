from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """
    h, w = shape
    steer_matrix_left = np.zeros(shape)
    
    # Create a matrix where values on the left side are positive (turn right)
    # and values on the right side are close to zero (minor correction)
    # This follows Braitenberg vehicle principles where the robot should turn away from the stimulus
    
    for i in range(w):
        # Calculate normalized position from left (-1) to right (1)
        normalized_pos = (i / w) * 2 - 1
        normalized_pos = normalized_pos * 0.5
        
        # Left side of image: turn right (positive values)
        if normalized_pos < 0:
            # Stronger steering the further left the marking is detected
            steer_matrix_left[:, i] = -normalized_pos * 2  # Positive values: turn right
        else:
            # Right side of image: minimal correction
            steer_matrix_left[:, i] = max(-normalized_pos * 0.5, 0)  # Small positive values
    
    # Gradient from bottom (max effect) to top (less effect)
    vertical_weight = np.linspace(0.1, 1, h).reshape(h, 1)
    steer_matrix_left = steer_matrix_left * vertical_weight
    
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """
    h, w = shape
    steer_matrix_right = np.zeros(shape)
    
    # Create a matrix where values on the right side are negative (turn left)
    # and values on the left side are close to zero (minor correction)
    
    for i in range(w):
        # Calculate normalized position from left (-1) to right (1)
        normalized_pos = (i / w) * 2 - 1
        normalized_pos = normalized_pos * 0.5

        # Right side of image: turn left (negative values)
        if normalized_pos > 0:
            # Stronger steering the further right the marking is detected
            steer_matrix_right[:, i] = -normalized_pos * 2  # Negative values: turn left
        else:
            # Left side of image: minimal correction
            steer_matrix_right[:, i] = min(-normalized_pos * 0.5, 0)  # Small negative values
    
    # Gradient from bottom (max effect) to top (less effect)
    vertical_weight = np.linspace(0.1, 1, h).reshape(h, 1)
    steer_matrix_right = steer_matrix_right * vertical_weight
    
    return steer_matrix_right


def detect_lane_markings(imgbgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    # OpenCV uses BGR by default, whereas matplotlib uses RGB, so we generate an RGB version for the sake of visualization
    imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY)

    # The image-to-ground homography associated with this image
    H = np.array([-4.137917960301845e-05, -0.00011445854191468058, -0.1595567007347241, 
                0.0008382870319844166, -4.141689222457687e-05, -0.2518201638170328, 
                -0.00023561657746150284, -0.005370140574116084, 0.9999999999999999])

    H = np.reshape(H,(3, 3))
    Hinv = np.linalg.inv(H)

    mask_ground = np.zeros(img.shape[:2], dtype=np.uint8)
    width = img.shape[1]
    horizon_points = []

    x_samples = np.linspace(0, width-1, 20)
    for x in x_samples:
        p_ground = np.array([x, 0, 1])
        p_image = Hinv @ p_ground
        p_image = p_image / p_image[2]  # Normalize by the homogeneous coordinate
        horizon_points.append((int(p_image[0]), int(p_image[1])))

    horizon_points = np.array(horizon_points)
    bottom_corners = np.array([(0, img.shape[0]-1), (width-1, img.shape[0]-1)])
    ground_polygon = np.vstack((horizon_points, bottom_corners))
    cv2.fillPoly(mask_ground, [ground_polygon], 255)

    if len(img.shape) == 3:
        mask_ground_3ch = np.stack([mask_ground, mask_ground, mask_ground], axis=2)
    else:
        mask_ground_3ch = mask_ground

    mask_ground = np.zeros(img.shape, dtype=np.uint8)
    height, width = img.shape

    # Extract the line at infinity from the original homography matrix
    a, b, c = H[2, :]
    
    # Calculate a single point on the horizon line (middle of image)
    middle_x = width // 2
    if abs(b) > 1e-10:  # Avoid division by zero
        horizon_y = int((-a*middle_x - c) / b)
        
        # Ensure horizon_y is within image bounds
        horizon_y = max(0, min(horizon_y, height-1))
        
        # Create a straight horizontal line mask
        # Everything below horizon_y is ground
        mask_ground[horizon_y:, :] = 1
    
    #mask_ground = cv2.bitwise_or(mask_ground, mask_ground_3ch)#cv2.bitwise_and(mask_ground, mask_ground_3ch)

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    sigma = 2 # CHANGE ME

    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    threshold = 100

    mask_mag = (Gmag > threshold)

    white_lower_hsv = np.array([10, 0, 190])         # CHANGE ME
    white_upper_hsv = np.array([180, 50, 255])   # CHANGE ME
    yellow_lower_hsv = np.array([15, 80, 150])        # CHANGE ME
    yellow_upper_hsv = np.array([35, 255, 255]) 

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    width = img.shape[1]
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    mask_left_edge = mask_ground * mask_left * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_white

    return mask_left_edge, mask_right_edge
