import math
import numpy as np, cv2
from Common.hough import accumulate, masking, select_lines


def houghLines(src, rho, theta, thresh):
    acc_mat = accumulate(src, rho, theta)
    acc_dst = masking(acc_mat, 7, 3, thresh)
    lines = select_lines(acc_dst, rho, theta, thresh)
    return lines


def draw_houghLines(src, lines, nline):
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    print(lines, nline)
    if lines is None:
        return None
    min_length = min(len(lines), nline)

    for i in range(min_length):
        rho, radian = lines[i, 0, 0:2]
        a, b = math.cos(radian), math.sin(radian)
        pt = (a * rho, b * rho)
        delta = (-1000 * b, 1000 * a)
        pt1 = np.add(pt, delta).astype('int')
        pt2 = np.subtract(pt, delta).astype('int')
        cv2.line(dst, tuple(pt1), tuple(pt2), (0, 255, 0), 2, cv2.LINE_AA)

    return dst


# 이미지 읽어오기
image = cv2.imread("12.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
if image is None:
    raise Exception("영상 파일 읽기 오류")

rho, theta = 1, np.pi / 180

# Canny Edge
canny2 = cv2.Canny(image, 100, 200)

# Hough Transform
lines2 = cv2.HoughLines(canny2, rho, theta, 80)

if lines2 is None:
    print('None Line')
else:
    dst2 = draw_houghLines(canny2, lines2, 7)
    cv2.imshow("HoughLine", dst2)

cv2.imshow("Original Image", image)

cv2.imshow("Canny Edge", canny2)

cv2.waitKey(0)
