import cv2
import os
import numpy as np

from config import *
from utils import resize_img, four_point_transform


def scan_document(img, img_name="scanned_img", save_steps=True, show_steps=True):
    orig_img = img.copy()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    top_hat_img = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)
    black_hat_img = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
    top_black_img = gray_img + top_hat_img - black_hat_img
    filtered_img = cv2.bilateralFilter(top_black_img, 15, 75, 75)
    kernel = np.ones((5,5),np.uint8)
    morph_img = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel, iterations=5)
    blurred_img = cv2.GaussianBlur(morph_img, (11, 11),0)
    edged_img = cv2.Canny(blurred_img, 30, 50) 
    ditaled_img = cv2.dilate(edged_img, (11, 11), iterations=5)
    ditaled_img_2 = cv2.dilate(ditaled_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, hierarchy = cv2.findContours(ditaled_img_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:5]
    cnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            cnt = approx
            x, y, w, h = cv2.boundingRect(cnt)
            break

    if cnt is None:
        cnt = max(contours, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)  
        cropped_img = orig_img[y:y+h, x:x+w]
    else:
        pts = cnt.reshape(4, 2)
        cropped_img = four_point_transform(orig_img, pts)

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)

    if save_steps:
        folder = f"{OUT_DIR}/{img_name}"
        if not os.path.exists(folder):
            os.makedirs(folder)

        cv2.imwrite(f"{folder}/1 - Original {img_name}.png", orig_img)
        cv2.imwrite(f"{folder}/2 - Gray {img_name}.png", gray_img)
        cv2.imwrite(f"{folder}/3 - Top Black {img_name}.png", top_black_img)
        cv2.imwrite(f"{folder}/4 - Filtered {img_name}.png", filtered_img)
        cv2.imwrite(f"{folder}/5 - Morph {img_name}.png", morph_img)
        cv2.imwrite(f"{folder}/6 - Blur {img_name}.png", blurred_img)
        cv2.imwrite(f"{folder}/7 - Edged {img_name}.png", edged_img)
        cv2.imwrite(f"{folder}/8 - Dilated {img_name}.png", ditaled_img)
        cv2.imwrite(f"{folder}/9 - Dilated {img_name} 2.png", ditaled_img_2)
        cv2.imwrite(f"{folder}/10 - Document found {img_name}.png", img)
        cv2.imwrite(f"{folder}/11 - Cropped {img_name}.png", cropped_img)


    if show_steps:
        cv2.imshow("Original image", orig_img)
        cv2.imshow("Gray image", gray_img)
        cv2.imshow("Top/Black image", top_black_img)
        cv2.imshow("Filtered image", filtered_img)
        cv2.imshow("Morph image", morph_img)
        cv2.imshow("Blur image", blurred_img)
        cv2.imshow("Edged image", edged_img)
        cv2.imshow("Dilated image", ditaled_img)
        cv2.imshow("Dilated image 2", ditaled_img_2)
        cv2.imshow("Cropped image", cropped_img)
        cv2.imshow("Scanned image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_img



img_list = sorted(os.listdir(IMG_DIR))
print(img_list)

for img_file in img_list:
    img_name = img_file.split(".")[0]
    img = cv2.imread(f"{IMG_DIR}/{img_file}")
    img = resize_img(img)
    scan_document(img, img_name=img_name, show_steps=False)

# img_file = img_list[0]
# img_name = img_file.split(".")[0]
# img = cv2.imread(f"{IMG_DIR}/{img_file}")
# img = resize_img(img)
# scan_document(img, img_name=img_name, show_steps=True, save_steps=True)