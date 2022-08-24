# import argparse
import cv2
import os

from config import IMG_DIR, OUT_DIR
from src.scan_document import scan_document


# ap = argparse.ArgumentParser(description="Document Scanner")
# ap.add_argument("-i", "--image", required=True, help = "path to the document image file")
# args = ap.parse_args()

# img_path = "/".join(args.image.split("/")[:-1])
# img_name = args.image.split("/")[-1].split(".")[0]
# img = cv2.imread(args.image)

# scanned_doc = scan_document(img, img_name=img_name, show_steps=False)
# cv2.imwrite(f"{img_path}/Document scanned - {img_name}.png", scanned_doc)


img_list = sorted(os.listdir(IMG_DIR))
print(img_list)

for img_file in img_list:
    img_name = img_file.split(".")[0]
    img = cv2.imread(f"{IMG_DIR}/{img_file}")
    scanned_doc = scan_document(img, img_name=img_name, show_steps=False)
    cv2.imwrite(f"{OUT_DIR}/{img_name}/Document scanned - {img_name}.png", scanned_doc)