from typing import Tuple

import numpy as np
import cv2

MOTOR_MAT_PATH_SIM = "/code/catkin_ws/src/braitenberg/packages/solution/motor_mat.png"
MOTOR_MAT_PATH_EDITOR = "../../packages/solution/motor_mat.png"

def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(MOTOR_MAT_PATH_SIM, cv2.IMREAD_COLOR_RGB)
    res = np.zeros(shape=shape, dtype="float32")
        
    if img is None:
        img = cv2.imread(MOTOR_MAT_PATH_EDITOR, cv2.IMREAD_COLOR_RGB)

    if img is not None:
        img = cv2.resize(img, (640, 480))
        r_channel, g_channel, b_channel = cv2.split(img)

        res_pos = np.divide(r_channel, 255)
        res_neg = np.divide(b_channel, 255)
        res += res_pos
        res -= res_neg
    
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(MOTOR_MAT_PATH_SIM, cv2.IMREAD_COLOR_RGB)
    res = np.zeros(shape=shape, dtype="float32")

    if img is None:
        img = cv2.imread(MOTOR_MAT_PATH_EDITOR, cv2.IMREAD_COLOR_RGB)

    if img is not None:
        img = cv2.resize(img, (640, 480))
        img = cv2.flip(img, 1)

        r_channel, g_channel, b_channel = cv2.split(img)

        res_pos = np.divide(r_channel, 255)
        res_neg = np.divide(b_channel, 255)
        res += res_pos
        res -= res_neg

    return res
