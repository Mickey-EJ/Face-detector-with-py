import cv2

# loading the pretrained data
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose an image to detect face in
img = cv2.imread('face.jpg')

# converting to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# printing the detected face coords
print(face_coordinates)

# Drawing rectangles on all face coords
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# showing the image after detection
cv2.imshow('Face detector program', img)

# wait key to pause code
cv2.waitKey()


print("Done")
