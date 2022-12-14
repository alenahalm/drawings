import numpy as np
import cv2 as cv

cam = cv.VideoCapture('video.mp4')

_, past_frame = cam.read()
_, prev_frame = cam.read()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    diff1 = cv.absdiff(prev_frame, frame)
    diff2 = cv.absdiff(past_frame, prev_frame)
    prev_frame = past_frame
    past_frame = frame
    movement1 = cv.threshold(diff1, 25, 255, cv.THRESH_BINARY)[1]
    movement1 = cv.erode(movement1, None)
    movement2 = cv.threshold(diff2, 25, 255, cv.THRESH_BINARY)[1]
    movement2 = cv.erode(movement2, None)

    if movement1.sum() > 1000 or movement2.sum() > 1000:
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 125, 255, 0)
    thresh = 255 - thresh
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    circles = 0
    line = 0
    for contour in contours:
        (x, y), rad = cv.minEnclosingCircle(contour)
        area = cv.contourArea(contour)
        p = cv.arcLength(contour, True)
        if area / (np.pi * rad * rad) > 0.84:
            circles += 1
        if area > 0 and p / area > 0.55:
            line += 1
    
    cv.putText(frame, f"My picture: {circles == 2 and line == 1}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))
    cv.imshow('frame', frame)

    k = cv.waitKey(10)
    if k > 0:
        if chr(k) == 'b':
            break