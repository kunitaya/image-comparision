from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import os
import datetime

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="First image")
ap.add_argument("-s", "--second", required=True, help="Second image")

args = vars(ap.parse_args())

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

if not os.path.isdir('output_' + datetime.datetime.now().strftime('%m-%d-%Y') + '_' + datetime.datetime.now().strftime('%H-%M-%S')):
    os.mkdir('output_' + datetime.datetime.now().strftime('%m-%d-%Y') + '_' + datetime.datetime.now().strftime('%H-%M-%S'))

if score < 0:
    percent = ((score * 100) / 2) - 50
    print("Similarity: %s" % percent + """%""")
else:
    percent = round(((score * 100) / 2) + 50, 2)
    print("Similarity: %s" % percent + """%""")
    text_file = open(os.path.join('output_' + datetime.datetime.now().strftime('%m-%d-%Y') + '_' + datetime.datetime.now().strftime('%H-%M-%S'), 'Similarity.txt'), 'a')
    text_file.write("Similarity: %s" % percent + """%""")
    text_file.close()

cv2.imwrite(os.path.join('output_' + datetime.datetime.now().strftime('%m-%d-%Y') + '_' + datetime.datetime.now().strftime('%H-%M-%S'), 'first_image_file.png'), imageA)
cv2.imwrite(os.path.join('output_' + datetime.datetime.now().strftime('%m-%d-%Y') + '_' + datetime.datetime.now().strftime('%H-%M-%S'), 'second_image_file.png'), imageB)
cv2.waitKey(0)
