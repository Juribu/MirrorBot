import cv2
from deepface import DeepFace
import time
import numpy as np
import os
import datetime
import pandas as pd 

class RealTimeFaceEmotionRecognition:
    def __init__(self,
                 prototxt_path,
                 model_path,
                 face_recognition_model='VGG-Face',
                 db_path=None,
                 frame_resize_factor=1.0):
        
        """
        Initialize the RealTimeFaceEmotionRecognition object.
        
        Parameters:
        - prototxt_path: Path to the prototxt file for the face detection model.
        - model_path: Path to the model file for the face detection model.
        - face_recognition_model: The face recognition model to use (default: 'VGG-Face').
        - db_path: Path to the directory to store the face database (default: None).
        - frame_resize_factor: Factor to resize the frame for faster processing (default: 1.0).
        """
        
        # Load face detection model
        self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.face_recognizer = DeepFace.build_model(face_recognition_model)
        self.frame_resize_factor = frame_resize_factor
        
        # Data structure to store recognized faces and their emotions
        self.face_db = {}
        self.face_id_counter = 0
        self.db_path = self._create_db_path()
        
        # Initialize variables for FPS calculation
        self.prev_tick = cv2.getTickCount()
        self.fps = 0
        
        # Initialize video capture
        self.cap=cv2.VideoCapture(0)
        
    def is_database_empty(self):
        """Checks if the face database directory is empty."""
        return len(os.listdir(self.db_path)) == 0
    
    def _create_db_path(self):
        """ Create a directory to store the face database. """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        db_path = f"face_database_{timestamp}/"
        os.makedirs(db_path)
        return db_path

    def detect_faces_dnn(self, frame, conf_threshold=0.7):
        """
        Detect faces in an image using the preloaded DNN model.
        
        Parameters:
        - frame: The input image.
        - conf_threshold: Confidence threshold for filtering detections.
        
        Returns:
        - faces: A list of bounding boxes for detected faces.
        """
        
        (h, w) = frame.shape[:2]
        
        # Prepare the blob from the image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        # Pass the blob through the network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > conf_threshold:
                # Compute the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x_start, y_start, x_end, y_end) = box.astype("int")
                
                # Ensure the bounding boxes are within the frame dimensions
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(w - 1, x_end)
                y_end = min(h - 1, y_end)
                
                # Append the face coordinates
                faces.append((x_start, y_start, x_end - x_start, y_end - y_start))
                
        return faces
    
    def recognize_face(self, face_roi):
        """
        Recognize the face in the given face ROI.
        
        Parameters:
        - face_roi: The face region of interest.
        
        Returns:
        - face_id: The ID of the recognized face.
        """
        
        # Check if the database has any faces
        if self.is_database_empty():
            print("Face database is empty. Saving new face.")
            self.face_id_counter += 1
            new_face_id = f'face_{self.face_id_counter}'
            cv2.imwrite(f'{self.db_path}{new_face_id}.jpg', face_roi)
            return new_face_id

        # Try to find the face in the database
        result = DeepFace.find(
            img_path=face_roi, 
            db_path=self.db_path, 
            enforce_detection=False, 
            silent=True, 
            threshold=0.6
        )
        
        # Check if the result contains any recognized faces
        if len(result) > 0 and not result[0].empty:
            # Extract the 'identity' field from the first result row
            face_identity = result[0]['identity'].iloc[0]
            face_id = os.path.splitext(os.path.basename(face_identity))[0]
            return face_id
        else:
            # If the face is new, assign a new ID and save the image
            self.face_id_counter += 1
            new_face_id = f'face_{self.face_id_counter}'
            cv2.imwrite(f'{self.db_path}{new_face_id}.jpg', face_roi)
            print(f"New face detected, saved as {new_face_id}")
            return new_face_id
        
                
    def process_frame(self, frame):
        """
        Process a single frame for face detection and recognition, analyze emotions and update FPS.
        
        Parameters:
        - frame: The input frame.
        
        Returns:
        - frame: The processed frame with bounding boxes and labels.
        """
        
        # Resize and flip frame for consistency
        frame = cv2.resize(frame, (0, 0), fx=self.frame_resize_factor, fy=self.frame_resize_factor)
        frame = cv2.flip(frame, 1)
        
        tick = cv2.getTickCount()
        # Detect faces
        faces = self.detect_faces_dnn(frame)
        
        # Recognize faces
        for (x, y, w, h) in faces:
            # Extract face ROI for recognition and emotion analysis
            face_roi = frame[y:y + h, x:x + w]
            
            # Recognize face
            face_id = self.recognize_face(face_roi)
            
            # Perform emotion analysis
            emotion_result = DeepFace.analyze(
                face_roi, 
                actions=['emotion'], 
                enforce_detection=False
            )
            emotion = emotion_result[0]['dominant_emotion']
            
            # Get current timestamp
            timestamp = time.strftime('%H:%M:%S')
            
            # Update face database
            self.face_db[face_id] = {
                'emotion': emotion,
                'timestamp': timestamp
            }
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{face_id} - {emotion}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Calculate the FPS
        time_diff = (tick - self.prev_tick) / cv2.getTickFrequency()
        self.prev_tick = tick  
        self.fps = 1.0 / time_diff
        
        cv2.putText(frame, f'FPS: {self.fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """
        Start the real-time face emotion recognition process.
        """
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame)
            
            cv2.imshow('Real-time Face Recognition & Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
    def save_results(self, csv_filename='emotion_log.csv'):
        """
        Save the face database to a CSV file.
        
        Parameters:
        - csv_filename: The name of the CSV file to save the face database (default: 'emotion_log.csv').
        """
        
        face_data = []
        for face_id, data in self.face_db.items():
            face_data.append([face_id, data['timestamp'], data['emotion']])
        
        df = pd.DataFrame(face_data, columns=['Face ID', 'Timestamp', 'Emotion'])
        df.to_csv(csv_filename, index=False)
        
        print(f"Emotion log saved to {csv_filename}")

        
if __name__ == "__main__":
    prototxt_path = '/Users/jidongzheng/Desktop/MirrorBot/FacialEmotion/deploy.prototxt.txt'
    model_path = '/Users/jidongzheng/Desktop/MirrorBot/FacialEmotion/res10_300x300_ssd_iter_140000.caffemodel'
    
    face_recognition_model = 'VGG-Face'
    emotion_recognition_model = 'Emotion'
    
    db_path = 'face_database/'
    
    face_emotion_recognizer = RealTimeFaceEmotionRecognition(
        prototxt_path, model_path, 
        face_recognition_model,
        db_path
    )
    
    face_emotion_recognizer.run()
    face_emotion_recognizer.save_results()