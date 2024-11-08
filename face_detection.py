# install opencv "pip install opencv-python" 
import cv2 

# distance from camera to object(face) measured 
# centimeter 
Known_distance = 76.2

# width of face in the real world or Object Plane 
# centimeter 
Known_width = 14.3

# Colors 
GREEN = (0, 255, 0) 
RED = (0, 0, 255) 
WHITE = (255, 255, 255) 
BLACK = (0, 0, 0) 

# defining the fonts 
fonts = cv2.FONT_HERSHEY_COMPLEX 

# face detector object 
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# focal length finder function 
def focal_length_finder(measured_distance, real_width, width_in_rf_image): 

	# finding the focal length 
	focal_length = (width_in_rf_image * measured_distance) / real_width 
	return focal_length 

# distance estimation function 
def distance_finder(Focal_Length, real_face_width, face_width_in_frame): 

	distance = (real_face_width * Focal_Length)/face_width_in_frame 

	# return the distance 
	return distance 


def face_data(image): 

	# returns face data on the first 2 faces in frame
	# converting color image to gray scale image 
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

	# detecting face in the image 
	faces = face_detector.detectMultiScale(gray_image, 1.3, 5) 

	# looping through the faces detect in the image 
	# getting coordinates x, y , width and height 

	face_widths = [0,0]

	for i in range(min(len(faces),2)):

		(x,y,h,w) = faces[i]

		# draw the rectangle on the face 
		cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2) 
		cv2.putText(image, 'Face ' + str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

		# getting face width in the pixels 
		face_widths[i] = w

	# return the face width in pixel 
	return face_widths 


# reading reference_image from directory 
ref_image = cv2.imread("ref_image.jpeg") 

# find the face width(pixels) in the reference_image 
ref_image_face_widths = face_data(ref_image)

# get the focal by calling "focal_length_finder" 
# face width in reference(pixels), 
# Known_distance(centimeters), 
# known_width(centimeters) 
focal_length = focal_length_finder( 
	Known_distance, Known_width, ref_image_face_widths[0]) 

# initialize the camera object so that we 
# can get frame from it 
cap = cv2.VideoCapture(0) 

# looping through frame, incoming from 
# camera/video 
while True: 

	# reading the frame from camera 
	_, frame = cap.read() 

	# calling face_data function to find 
	# the width of face(pixels) in the frame 
	face_widths_in_frame = face_data(frame) 

	# check if the face is zero then not 
	# find the distance 

	face_width_in_frame_0 = face_widths_in_frame[0]
	face_width_in_frame_1 = face_widths_in_frame[1]

	if face_width_in_frame_0 != 0 or face_width_in_frame_1 != 0: 
		
		# finding the distance by calling function 
		# Distance finder function need 
		# these arguments the Focal_Length, 
		# Known_width(centimeters), 
		# and Known_distance(centimeters) 
		Distance_0 = distance_finder( 
			focal_length, Known_width, face_width_in_frame_0) 
		
		Distance_1 = distance_finder( 
			focal_length, Known_width, face_width_in_frame_1) 

		# draw line as background of text 
		cv2.line(frame, (30, 30), (330, 30), RED, 32) 
		cv2.line(frame, (30, 30), (330, 30), BLACK, 28) 

		cv2.line(frame, (30, 70), (330, 70), RED, 32) 
		cv2.line(frame, (30, 70), (330, 70), BLACK, 28) 

		# Drawing Text on the screen 
		cv2.putText( 
			frame, f"Distance, face 0: {round(Distance_0,2)} CM", (30, 35), 
		fonts, 0.6, GREEN, 2) 

		cv2.putText( 
			frame, f"Distance, face 1: {round(Distance_1,2)} CM", (30, 75), 
		fonts, 0.6, GREEN, 2) 

	# show the frame on the screen 
	cv2.imshow("frame", frame) 

	# quit the program if you press 'q' on keyboard 
	if cv2.waitKey(1) == ord("q"): 
		break

# closing the camera 
cap.release() 

# closing the windows that are opened 
cv2.destroyAllWindows() 
