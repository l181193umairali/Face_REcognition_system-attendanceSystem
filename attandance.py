import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
from datetime import date
# Declaring the Path
path = 'ImagesAttandance'
# making a list to store each and every student
Images = []
classNames = []
# giving directory to my list to get all images
myList = os.listdir(path)
print(myList)

# Now print the list without the extension
# get images one by one
for cl in myList:
    curImages = cv2.imread(f'{path}/{cl}')
    Images.append(curImages)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)


def findencodings(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


nameList = []


def markAttandance(name):
    # we want to read and write at the same time
    with open('attandance.csv', 'r+') as f:
        myDataList = f.readline()
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            today = date.today()
            f.writelines(f'\n{name},{dtString},{today}')
            nameList.append(name)


encodeListKnown = findencodings(Images)
print('Encoding complete')

# 3rd step
# To compare  each train image with the original image
# for this we have to open web cam so that it can catpure the image and compare it with already present image
# 0 means assign each image a unique id

capture = cv2.VideoCapture(0)
nameList2 = []
while True:
    success, img = capture.read()
    # (0,0) means pixel size which is not specifrid yet
    # we scaled down  our image to 1/4th the original size
    imS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imS = cv2.cvtColor(imS, cv2.COLOR_BGR2RGB)
    # We can have multiple object in a pic, in order to find location of a face we have to write this.

    faceCurrFrame = face_recognition.face_locations(imS)
    encodingCurrframe = face_recognition.face_encodings(imS, faceCurrFrame)
    # it will grab one face location from facecurrframe and grab encodinds of currframe from  encodingcurrrframe one by one
    # as we want them in same loop thats why we use zip

    for encodeface, faceLoc in zip(encodingCurrframe, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        fasDis = face_recognition.face_distance(encodeListKnown, encodeface)
        print(fasDis)
        matchIndex = np.argmin(fasDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            # in order to get a full rectangle we have to multiply by 4
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # we want to disolay the name at the bottom of rectangle
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            markAttandance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
