import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('faces.png')

# The image is in color, but we need it in grayscale
gray_scaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)

# Draw rectangles around the faces
for face_coordinate in face_coordinates:
    (x, y, w, h) = face_coordinate
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the faces spotted
cv2.imshow('Face Detector', img)
cv2.waitKey()