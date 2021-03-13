import cv2
import dlib
from playsound import playsound
from scipy.spatial import distance
import threading
import imutils

def total_function():	
	def calculate_EAR(eye):
		A = distance.euclidean(eye[1], eye[5])
		B = distance.euclidean(eye[2], eye[4])
		C = distance.euclidean(eye[0], eye[3])
		ear_aspect_ratio = (A+B)/(2.0*C)
		return ear_aspect_ratio

	def play_alarm(path):
		playsound(path)

	alarm_file = "alarm.WAV"

	EAR_THRESHOLD = 0.26
	EYE_AR_CONSEC_FRAMES = 26

	COUNTER = 0
	ALARM_PLAYING = False

	vid = cv2.VideoCapture(0)

	face_detector = dlib.get_frontal_face_detector()
	landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	while True:
		_, frame = vid.read()
		frame = imutils.resize(frame, width=1024)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = face_detector(gray)
		for face in faces:

			face_landmarks = landmark_predictor(gray, face)
			leftEye = []
			rightEye = []

			for n in range(36,42):
				x = face_landmarks.part(n).x
				y = face_landmarks.part(n).y
				leftEye.append((x,y))
				next_point = n+1
				if n == 41:
					next_point = 36
				x2 = face_landmarks.part(next_point).x
				y2 = face_landmarks.part(next_point).y
				cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

			for n in range(42,48):
				x = face_landmarks.part(n).x
				y = face_landmarks.part(n).y
				rightEye.append((x,y))
				next_point = n+1
				if n == 47:
					next_point = 42
				x2 = face_landmarks.part(next_point).x
				y2 = face_landmarks.part(next_point).y
				cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

			left_ear = calculate_EAR(leftEye)
			right_ear = calculate_EAR(rightEye)

			EAR = (left_ear+right_ear)/2
			EAR = round(EAR,2)
			if EAR < EAR_THRESHOLD:
				COUNTER += 1

				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					if not ALARM_PLAYING:
						ALARM_PLAYING = True

						t1 = threading.Thread(target=play_alarm, args=(alarm_file,))
						t1.deamon = True
						t1.start()

					cv2.putText(frame,"DROWSY",(140,400),
						cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),3)
					# cv2.putText(frame,"Are you Sleepy?",(20,400),
					# 	cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
					print("Drowsy")
			else:
				COUNTER = 0
				ALARM_PLAYING = False    	
			print(EAR)
			cv2.putText(frame,"EAR: {:.2f}".format(EAR),(260,30),
				cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

		cv2.imshow("Driver Drowsiness Detector", frame)

		#Pressing the escape key will close the window. 27 is the ASCII value
		#for the escape key
		key = cv2.waitKey(1)
		if key == 27:
			break
	vid.release()
	cv2.destroyAllWindows()