import os
import cv2
import numpy as np
import face_recognition as fr
from datetime import datetime
import mediapipe as mp
import pyttsx3

Header = "\nG.O.D.S.E.Y.E.S (Guided Object Detection and Surveillance for Enhanced Yielding Estimation System)\n"
print(Header)

haraya_engine = pyttsx3.init()
voices = haraya_engine.getProperty('voices')
haraya_engine.setProperty('voice', voices[0].id)

def speak(text):
    haraya_engine.say(text)
    haraya_engine.runAndWait()
    
def Play_Prompt_Sound():
    from playsound import playsound
    mp3_path = U"prompt1.mp3"
    playsound(mp3_path)

def Locate_NameHA(name):
    Honorific_Address = ""
    Male_Names = ["Gianne Bacay",
                "Earl Jay Tagud",
                "Gemmuel Balceda",
                "Mark Anthony Lagrosa",
                "Klausmieir Villegas",
                "CK Zoe Villegas", 
                "Pio Bustamante",
                "Rolyn Morales",
                "Alexander Villasis",
                "Meljohn Aborde",
                "Kimzie Torres",
                "Vonn Cedric Escodero"]

    Female_Names = ["Kleinieir Pearl Kandis Bacay",
                    "Princess Viznar",
                    "Nichi Bacay",
                    "Roz Waeschet Bacay",
                    "Killy Obligation",
                    "Jane Rose Bandoy",
                    "Ilyn Petalcorin"]

    if name in Male_Names:
        Honorific_Address = "Sir"
    elif name in Female_Names:
        Honorific_Address = "Ma'am"
    return Honorific_Address


def YOLO_Object_Detection(frame):
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        return output_layers
    output_layers = get_output_layers(net)
    classes = [line.strip() for line in open("coco.names")]

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Parsing through detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype('int')
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw the rectangle on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def Face_Pose_Recognition_System():
    def ClearCSV():
        import csv
        file = open("database.csv", "r")
        csvr = csv.reader(file)
        namelist = []
        Header = f'Name, Time'
        Header = Header.split(',')
        namelist.insert(0, Header)
        file = open("database.csv", "w", newline='')
        csvr = csv.writer(file)
        csvr.writerows(namelist)
        file.close()

    def get_encoded_faces():
        encoded = {}
        for dirpath, dnames, fnames in os.walk("./faces"):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file("faces/" + f)
                    encoding = fr.face_encodings(face)[0]
                    encoded[f.split(".")[0]] = encoding
        return encoded

    def SaveToDatabase(name):
        if name == "Unknown":
            response = "Unknown face was detected"
            print(response)
            speak(response)
        else:
            with open("database.csv", 'r+') as attendance:
                MyDatalist = attendance.readlines()
                NameList = set()
                for line in MyDatalist:
                    entry = line.split(',')
                    NameList.add(entry[0])
                if name not in NameList:
                    now = datetime.now()
                    Time = now.strftime('%I:%M %p')
                    attendance.writelines(f'\n{name}, {Time}')
                    NameHA = Locate_NameHA(name)
                    response = f"{NameHA} {name} was detected!"
                    print(response)
                    speak(response)
    
    def CountFaces(frame, face_locations):
        count = len(face_locations)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, f"Faces: {count}", (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    def resize(frame, size):
        width = int(frame.shape[1] * size)
        height = int(frame.shape[0] * size)
        dimension = (width, height)
        return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    ClearCSV()

    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        Play_Prompt_Sound()
        response = "System is now online."
        print(response)
        speak(response)
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = resize(frame, 0.70)
            frame = cv2.flip(frame, 1)
            frame = YOLO_Object_Detection(frame)
            frame_height, frame_width, _ = frame.shape
            results = holistic.process(frame)
            face_locations = fr.face_locations(frame)
            unknown_face_encodings = fr.face_encodings(frame, face_locations)
            face_names = []
            for face_encoding in unknown_face_encodings:
                matches = fr.compare_faces(faces_encoded, face_encoding)
                name = "Unknown"
                face_distances = fr.face_distance(faces_encoded, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=results.face_landmarks,
                        connections=mp_holistic.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.putText(frame, name, (left - 30, top - 30), font + 1, 1, (255, 255, 255), 1)
                    SaveToDatabase(name)

            CountFaces(frame, face_locations)

            # Draw landmarks for face, hands, and pose
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.left_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.right_hand_landmarks,
                connections=mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            cv2.imshow('G.O.D.S.E.Y.E.S (Guided Object Detection and Surveillance for Enhanced Yielding Estimation System)', frame)
            if cv2.waitKey(30) & 0xff == 27:
                cap.release()
                cv2.destroyAllWindows()
                return face_names

    cap.release()
    cv2.destroyAllWindows()

print("Initializing...")
speak("Initializing GODSEYES")
Face_Pose_Recognition_System()
Play_Prompt_Sound()
#______________________________python yolov3.py

