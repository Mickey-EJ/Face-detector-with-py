import cv2

# loading the pretrained data
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose video source to detect face in
webcam = cv2.VideoCapture(0)

# Iterate over all frames
while True:
    # read current frame
    successful_frame_read, frame = webcam.read()

    # converting to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Drawing rectangles on all face coords
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # showing the image after detection
    cv2.imshow('Face detector program', frame)

    # Press Q to quit!
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


#flush the webcam object   
webcam.release()

print("Ran fine")
