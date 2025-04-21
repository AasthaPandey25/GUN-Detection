import numpy as np
import cv2
import imutils
import datetime

# Load the gun detection cascade file
gun_cascade = cv2.CascadeClassifier('cascade.xml')  # Ensure the path to cascade.xml is correct
camera = cv2.VideoCapture(0)  # Start capturing video from the webcam

firstFrame = None
gun_exist = False

while True:
	ret, frame = camera.read()  # Read the video frame
	if frame is None:
		break

	# Resize the frame for easier processing
	frame = imutils.resize(frame, width=500)

	# Convert the frame to grayscale (needed for gun detection)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect guns in the frame
	gun = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))
	gun_exist = len(gun) > 0  # Check if any guns were detected

	# Draw rectangles around detected guns
	for (x, y, w, h) in gun:
		frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

	# Save the first frame for reference
	if firstFrame is None:
		firstFrame = gray
		continue

	# Overlay the current timestamp on the frame
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
				(10, frame.shape[0] - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.35, (0, 0, 255), 1)

	# Display the video feed in a window
	if gun_exist:
		print("Guns detected")
		cv2.putText(frame, "Gun Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	else:
		cv2.putText(frame, "No Gun Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

	cv2.imshow("Gun Detection - Live", frame)  # Display the frame in an OpenCV window

	# Exit the loop when the 'q' key is pressed
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()