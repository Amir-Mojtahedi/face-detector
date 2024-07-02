import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    successfull_frame_read, frame = webcam.read()
    
    # The image is in color, but we need it in grayscale
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(gray_scaled_img)
    
    # Draw rectangles around the faces
    for face_coordinate in face_coordinates:
        (x, y, w, h) = face_coordinate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with the faces spotted
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)
    
    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break



