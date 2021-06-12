import json
import cv2, math

# json 생성
def make_json_file(pill_shape, pill_text, pill_line=False):
    # json 기본 구조 설정
    drugData = {
        "drug_name" : "",
        "drug_type" : "",
        "drug_shape" : "",
        "drug_color" : "",
        "drug_line" : "없음"
    }
    drugData['drug_shape'] = pill_shape # 알약 모양
    drugData["drug_name"] = pill_text # 알약 문자
    # drugData["drug_line"] = pill_line
    try:
        # 알약 특징 정보를 가진 json 파일 생성
        with open('pill_data.json', 'w') as pill_data:
            json.dump(drugData, pill_data, indent='\t')
    except:
        print("error")

# crop한 text영역 이미지 회전 (croped img, polys=영역 좌표, 중심좌표)
def rotate_img(img, polys, center):
    center = tuple(map(int, center))
    # 좌상단(polys[0]), 좌하단(polys[3])
    # 좌하단을 기준으로 절대각도 확인
    x1, y1 = polys[0]
    x3, y3 = polys[2]
    x4, y4 = polys[3]
    dx, dy = x4 - x3, y4 - y3
    dhx, dhy = x4 - x1, y4 - y1
    angle = math.degrees(math.atan2(dy, dx))
    # text영역 크기 지정
    imgW = int(math.sqrt((dx**2)+(dy**2)))
    imgH = int(math.sqrt((dhx**2)+(dhy**2)))

    if dx < 0:
        angle += 180

    height, width, channels = img.shape
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # atan2로 구한 각도에 맞추어 회전
    img_rot = cv2.warpAffine(img, M, (width, height))
    # text 영역 중심을 기준으로 이미지 crop
    img_crop = cv2.getRectSubPix(img_rot, (imgW,imgH), center)
    if imgH > (imgW*1.3):
        img_crop = cv2.rotate(img_crop, cv2.ROTATE_90_CLOCKWISE)

    return img_crop