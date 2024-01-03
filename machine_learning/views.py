from django.shortcuts import render, redirect,HttpResponse
import cv2
import json
from django.core.serializers import serialize
import numpy as np
import speech_recognition as sr
import subprocess
from datetime import datetime
from deepface import DeepFace
from .models import *
import ast
import time
from pdfminer.high_level import extract_text
from io import BytesIO
from pydub import AudioSegment  # Import AudioSegment for voice modulation analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer  # Import SentimentIntensityAnalyzer for sentiment analysis
import mediapipe as mp
from math import degrees, acos
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import librosa
from tensorflow import keras
import dlib
import tensorflow as tf
import imutils 
from scipy.spatial import distance as dist 
from imutils import face_utils 
import pickle
import sys
import os
from voice_emotion import extract_feature
import re
from tensorflow.keras.models import load_model
from pydub.silence import split_on_silence
import math as m
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import *
from rest_framework.parsers import FormParser, MultiPartParser
import uuid
from .body_posture_detection import body_posture as detect_body_posture
from moviepy.editor import VideoFileClip, AudioFileClip
import pyaudio
import threading
import wave
from django.core.files import File
from django.core.files.base import ContentFile
from PIL import Image
from django.http import JsonResponse
from django.urls import reverse


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


detector = dlib.get_frontal_face_detector() 
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(project_root_dir, 'shape_predictor_68_face_landmarks.dat')
landmark_predict = dlib.shape_predictor(folder_path)

def preprocess_frame(frame):
    # Resize to match the input shape of your model
    resized_frame = cv2.resize(frame, (48, 48))
    # Convert to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)   
    # Normalize pixel values to be between 0 and 1
    normalized_frame = gray_frame / 255.0   
    # Expand dimensions to match the input shape of your model
    preprocessed_frame = np.expand_dims(normalized_frame, axis=-1)   
    return np.expand_dims(preprocessed_frame, axis=0)

def calculate_EAR(eye):  
    # calculate the vertical distances 
    y1 = dist.euclidean(eye[1], eye[5]) 
    y2 = dist.euclidean(eye[2], eye[4]) 
  
    # calculate the horizontal distance 
    x1 = dist.euclidean(eye[0], eye[3]) 
  
    # calculate the EAR 
    EAR = (y1+y2) / x1 
    return EAR 


def scan_face(request):
    video_path = None

    if request.method == 'POST':
        video_file = request.FILES['video']
        if video_file.size > 10 * 1024 * 1024:
            return render(request, 'upload.html', {
                "message": "Video size should be up to 10MB.",
                "tag": "danger",
            })
        # Create a temporary file to store the uploaded video
        temp_video_path = f"{video_file.name}"
        with open(temp_video_path, 'wb') as temp_file:
            for chunk in video_file.chunks():
                temp_file.write(chunk)

        # Use moviepy to get the video duration
        clip = VideoFileClip(temp_video_path)
        duration = clip.duration

        # Check if video duration is greater than 30 seconds
        if duration > 30:
            os.remove(f"{video_file}")
            return render(request, 'upload.html', {
                "message": "Video duration should be less than 30 seconds.",
                "tag": "danger",
            })

        try:
            video_data = VideoRecognition.objects.get(name=f"{video_file}")
        except VideoRecognition.DoesNotExist:
            video_data = None

        if video_data is None:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
            try:
                audio_file_path = generate_audio_file(f"{video_file}")
                print(audio_file_path)
                language_sentiment_analysis, voice_modulation_data,energy_category,filler_words,words_list,greeting_words = analyze_language_and_voice(audio_file_path)
                # Get speech rate
                wpm = calculate_speech_rate(audio_file_path)
                speech_rate = round(wpm,2)
                monotone = voice_monotone(audio_file_path)
                pauses = detect_voice_pauses(audio_file_path)
                language_analysis = language_sentiment_analysis["sentiment"]
                energy_level,energy_score = energy_category
                voice_modulation = {"pitch":voice_modulation_data["pitch"],"modulation_rating":voice_modulation_data["modulation_rating"]}

                if len(greeting_words) > 0:
                    greeting = "Greeting included"
                else:
                    greeting = None

                emo = voice_emotion(audio_file_path)
                # Convert NumPy array to Python list
                voice_emo = emo.tolist() if isinstance(emo, np.ndarray) else emo

            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")


            cap = cv2.VideoCapture(f"{video_file}")

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Open a video capture object
            output_filename = f"output_{video_file}"
            video_output = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)


            # Variables 
            blink_thresh = 0.45
            succ_frame = 2
            count_frame = 0

            blinks_per_minute = 0
            current_second_start_time = time.time()
            # Example: Process every 11rd frame
            frame_skip = 11
            frame_count = 0  # Initialize the frame count

            eye_contact = None
            hand_move = None
            eye_bling = None
            b_confidence = None
            thanks = None
            greet_gesture = None
            # Initialize variables for tracking time
            start_time = time.time()
            total_detected_time = 0
            total_not_detected_time = 0

            # Initialize variables for Body Posture
            good_posture_time = 0
            bad_posture_time = 0

            hand_movement_count = 0
            none_hand_movement_count = 0
            emotion_change = 0
            emotion_not_detected = 0
            eye_contact_detect = 0
            eye_not_contact = 0
            body_confidence_count = 0
            not_body_confidence_count = 0

            # List of possible emotions
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            # Initialize a dictionary to store emotion counts
            emotion_counts = {emotion: 0 for emotion in emotions}

            # Create a VideoRecognition object and associate it with the video_frame
            video_recognition = VideoRecognition.objects.create(name=str(video_file))

            while True:
                # Capture frames.
                success, image = cap.read()
                if not success:
                    print("Null Frames")
                    break
                # Specify the output thumbnail filename
                file = str(video_file)
                thumb = file.replace(".mp4", "")
                thumbnail_filename = f"thumbnail_{thumb}.jpg"

                # Save the first frame as a thumbnail image
                cv2.imwrite(thumbnail_filename, image)

                current_time = frame_count / fps
                print("video duration for every frame",current_time)
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue  # Skip frames
                try:                   
                    posture = detect_body_posture(image, fps)
                    good_time, bad_time = posture
                    print("Good Posture Time:", good_time)
                    print("Bad Posture Time:", bad_time)
                    if good_time > 0:
                        good_posture_time += good_time
                        save_detected_frame(video_recognition, "good_posture", image,frame_count,current_time)

                    else:
                        bad_posture_time += bad_time
                        save_detected_frame(video_recognition, "bad_posture", image,frame_count,current_time)

                except Exception as e:
                    print(e)
                gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Detect faces in the frame
                faces = detector(gray_frame)

                if len(faces) > 0:
                    total_detected_time += time.time() - start_time
                    start_time = time.time()
                    save_detected_frame(video_recognition, "face_detected", image,frame_count,current_time)
                    for face in faces:
                        # Get facial landmarks
                        landmarks = landmark_predict(gray_frame, face)

                        # Draw circles around each landmark point
                        for n in range(0, 68):
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

                        # Draw rectangle around the face
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                else:
                    total_not_detected_time += time.time() - start_time
                    start_time = time.time()
                    save_detected_frame(video_recognition, "face_not_detected", image,frame_count,current_time)
                
                frame = cv2.flip(image, 1)
                # Emotion Changes Detection
                predicted_emotion = get_emotion_change(face_cascade,image)

                # Update the count for the detected emotion
                if predicted_emotion in emotion_counts:
                    emotion_counts[predicted_emotion] += 1
                else:
                    pass

                
                if predicted_emotion is not None:
                    emotion_change += 1
                    cv2.putText(image, predicted_emotion, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    save_detected_frame(video_recognition, f"{predicted_emotion}", image,frame_count,current_time)
                else:
                    emotion_not_detected += 1
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image with MediaPipe Hands
                # Hand Movement, Thanks Geesture and body confidence detection code functions *****************
                greeting_gesture = hand_greeting_gesture(frame)
                hand_track = hand_movement(image)
                thanks_gesture = get_thanks_gesture(image)
                confidence = body_confidence(image)
                # Convert the RGB image to BGR.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if greeting_gesture == "Namaste" or greeting_gesture == "Hi/Hello":
                    greet_gesture = "Greeting gesture included"
                    cv2.putText(image, greeting_gesture, (20, 100), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
                    save_detected_frame(video_recognition, "greeting_gesture", image,frame_count,current_time)
                else:
                    save_detected_frame(video_recognition, "no_gesture", image,frame_count,current_time)

                if hand_track is not None:
                    hand_move = 'Hand Moving'
                    hand_movement_count += 1
                    hand_track,x,y = hand_track
                    cv2.putText(image, 'Hand Moving', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    save_detected_frame(video_recognition, "hand_moving", image,frame_count,current_time)
                else:
                    none_hand_movement_count += 1
                    save_detected_frame(video_recognition, "hand_not_moving", image,frame_count,current_time)

                if thanks_gesture is not None:
                    thanks_gesture,x,y = thanks_gesture
                    thanks = "Thanking gesture included"
                    cv2.putText(image, 'Thanks Gesture', (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    save_detected_frame(video_recognition, "thanks", image,frame_count,current_time)
                else:
                    save_detected_frame(video_recognition, "no_thanks", image,frame_count,current_time)

                if confidence == "Confident":
                    b_confidence = "Confident body posture"
                    body_confidence_count += 1
                    # Display the posture on the frame
                    cv2.putText(image, f"Posture: {confidence}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 51 ), 2)
                    save_detected_frame(video_recognition, "confident", image,frame_count,current_time)
                else:
                    not_body_confidence_count += 1
                    cv2.putText(image, f"Posture: {confidence}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 51 ), 2)
                    save_detected_frame(video_recognition, "unconfident", image,frame_count,current_time)

                # Eye Contact detection code start ***********
                eye_distance, x, y, w, h = eye_contact_detection(image, face_cascade, eye_cascade)
                eye_contact_threshold = 20  # Example threshold, you may need 
                # Check if eyes are horizontally aligned (within the threshold)
                if eye_distance < eye_contact_threshold:
                    eye_contact = 'Eye Contact'
                    eye_contact_detect += 1
                    cv2.putText(image, 'Eye Contact', (x, y + h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 233, 51), 2)
                    save_detected_frame(video_recognition, "eye_contact", image,frame_count,current_time)
                else:
                    eye_not_contact += 1
                    save_detected_frame(video_recognition, "not_contact", image,frame_count,current_time)

                # Eye Blinging detection code start *************
                blinging_detected = eye_blinging(image)
                if blinging_detected < blink_thresh: 
                    count_frame += 1  # incrementing the frame count 
                else: 
                    if count_frame >= succ_frame: 
                        blinks_per_minute += 1
                        cv2.putText(image, 'Blink Detected', (60, 50), 
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1) 
                    else: 
                        count_frame = 0

                elapsed_time = time.time() - current_second_start_time
                if elapsed_time >= 5:
                    if blinks_per_minute > 2:
                        eye_bling = "Blink more often"
                        save_detected_frame(video_recognition, "more_blinging", image,frame_count,current_time)
                        cv2.putText(image, f'{blinks_per_minute} Blinks in 5 Second', (60, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                    blinks_per_minute = 0
                    current_second_start_time = time.time()

                # Eye Blinging detection code start *************
                smile_detect, x, y, w, h = smile_detection(image, face_cascade, smile_cascade)
                if smile_detect > 0:
                    cv2.putText(image, 'Smiling', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                
                # Write the frame to the output video.
                video_output.write(image)
                # new_width = 500
                # new_height = 600
                # # Resize the image
                # resized_image = cv2.resize(image, (new_width, new_height))
                # # Display the frame.
                # cv2.imshow('Video', resized_image)
                # # Break the loop if 'q' key is pressed.
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
            # Release video capture and writer objects.
            cap.release()
            video_output.release()
            cv2.destroyAllWindows()
            
            # Find the emotion with the maximum count
            most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)

            # Analysis score detection code start ##################
            if good_posture_time > 0:
                posture_ratio = good_posture_time/(good_posture_time + bad_posture_time)
                if posture_ratio > 0.5:
                    body_posture = "Good Body Posture"
                else:
                    body_posture = "Bad Body Posture"
            else:
                posture_ratio = 0.0
                body_posture = None

            if hand_movement_count > 0:
                hand_move_ratio = hand_movement_count / (hand_movement_count + none_hand_movement_count) 
            else:
                hand_move_ratio = 0.0
 
            body_language_ratio = ((posture_ratio + hand_move_ratio)/2) * 100
            body_language_score = round(body_language_ratio,2)

            if emotion_change > 0:
                emotion_change_ratio = (emotion_change/(emotion_change + emotion_not_detected)) * 100
            else:
                emotion_change_ratio = 0.0

            if eye_contact_detect > 0:
                eye_contact_ratio = (eye_contact_detect/(eye_contact_detect + eye_not_contact)) * 100
            else:
                eye_contact_ratio = 0.0
            facial_expression_ratio = (emotion_change_ratio + eye_contact_ratio)/2
            facial_expression_score = round(facial_expression_ratio,2)

            if not filler_words:
                language_analysis_score = language_sentiment_analysis["sentiment_score_average"] * 100
                language_analysis_average = 0.0
            else:
                language_analysis_average = ((language_sentiment_analysis["sentiment_score_average"] + 1.0) / 2) * 100  # 1.0 is added for filler words average to get percentage
                language_analysis_score = round(language_analysis_average, 2)

            if body_confidence_count > 0:
                confidence_ratio = body_confidence_count/(body_confidence_count + not_body_confidence_count)
            else:
                confidence_ratio = 0.0
            
            body_confidence_score = round((confidence_ratio * 100),2)

            voice_modulation_score = round((energy_score + voice_modulation_data["percentage_modulation"])/2,2)

            total_len = total_detected_time + total_not_detected_time
            ratio = total_detected_time/total_len
            if ratio > 0.5:
                face_detected = "Appropriate Facial Detected."
            else:
                face_detected = "Appropriate Facial Not Detected."
            
            # Getting Total Video Ana;ysis Score ####################
            t_score = get_analysis_score(body_language_score,facial_expression_score,voice_modulation_score,body_confidence_score,language_analysis_score)
            try:
                # Update the fields with new data
                video_recognition.thumb_img = File(open(thumbnail_filename, 'rb'))
                video_recognition.analysis_score = t_score
                video_recognition.language_analysis = language_analysis
                video_recognition.voice_modulation_analysis = voice_modulation
                video_recognition.energy_level_analysis = energy_level
                video_recognition.video_file = video_file
                video_recognition.video_durations = duration
                video_recognition.word_per_minute = speech_rate
                video_recognition.filler_words_used = filler_words
                video_recognition.frequently_used_word = words_list
                video_recognition.voice_emotion = voice_emo
                video_recognition.confidence = b_confidence
                video_recognition.eye_bling = eye_bling
                video_recognition.hand_movement = hand_move
                video_recognition.eye_contact = eye_contact
                video_recognition.thanks_gesture = thanks
                video_recognition.greeting = greeting
                video_recognition.greeting_gesture = greet_gesture
                video_recognition.voice_tone = monotone
                video_recognition.voice_pauses = pauses
                video_recognition.appropriate_facial = face_detected
                video_recognition.body_posture = body_posture
                video_recognition.body_language_score = body_language_score
                video_recognition.facial_expression_score = facial_expression_score
                video_recognition.language_analysis_score = language_analysis_score
                video_recognition.voice_modulation_score = voice_modulation_score
                video_recognition.body_confidence_score = body_confidence_score
                video_recognition.facial_expression = most_frequent_emotion

                # Save the updated object
                video_recognition.save()
                
            except Exception as e:
                print(e)
                pass

            try:
                os.remove(f"{video_file}")
                os.remove(f"output_{video_file}")
                os.remove(audio_file_path)
                os.remove(thumbnail_filename)
            except:
                pass
            return redirect("analized_video_detail",video_recognition.id)
        else:
            os.remove(f"{video_file}")
            return render(request, 'upload.html', {
                "message": "Video already exists with this name and this video data shown below.",
                "tag": "info",
                'video_data': video_data,
            })

    return render(request, 'upload.html')

def save_detected_frame(video_recognition_obj, detected_data, image_frame, frame_count,current_time):
    # Convert the frame to a JPEG image
    pil_image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    frame_image = ContentFile(buffer.getvalue(), f"{detected_data}_frame_{video_recognition_obj.id}.jpg")

    # Assuming video_recognition_obj is an instance of VideoRecognition
    try:
        # Get or create the posture associated with the video
        posture, created = Posture.objects.get_or_create(video=video_recognition_obj, name=detected_data)

        # Get or create the DetectedFrames instance for that Posture
        detected_frame, created = DetectedFrames.objects.get_or_create(posture=posture)

        # Add the frame to the many-to-many relationship with the correct frame number
        frame, created = Frame.objects.get_or_create(image=frame_image, number=frame_count,current_time=current_time)
        detected_frame.frames.add(frame)

    except Exception as e:
        print(f"Error: {e}")
        return

  


def get_analysis_score(body_language_score,facial_expression_score,voice_modulation_score,body_confidence_score,language_analysis_score):
    average = (body_language_score + facial_expression_score + voice_modulation_score + body_confidence_score + language_analysis_score)/5
    
    return round(average,2)  # Invert the score to get a percentage


def eye_blinging(image):
    # Eye landmarks 
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 
    frame = imutils.resize(image, width=640) 
    # converting frame to gray scale to 
    # pass to detector 
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_faces = detector(img_gray) 
    for face in eye_faces:            
        # landmark detection 
        shape = landmark_predict(img_gray, face) 
        # converting the shape class directly 
        # to a list of (x,y) coordinates 
        shape = face_utils.shape_to_np(shape) 
        # parsing the landmarks list to extract 
        # lefteye and righteye landmarks--# 
        lefteye = shape[L_start: L_end] 
        righteye = shape[R_start:R_end] 
        # Calculate the EAR 
        left_EAR = calculate_EAR(lefteye) 
        right_EAR = calculate_EAR(righteye) 
        # Check if any EAR calculation is unsuccessful
        if left_EAR is None or right_EAR is None:
            return 0.0  # or any default value you prefer
        # Avg of left and right eye EAR 
        avg = (left_EAR + right_EAR) / 2
        return avg
    return 0.0  # Return default value if no face is found

def eye_contact_detection(image, face_cascade, eye_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    x, y, w, h = 0, 0, 0, 0  # Initialize these variables before the loop
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        
        # Eyes detection
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda eye: eye[0])
            
            # Extract the coordinates of the two eyes
            (eye1_x, eye1_y, eye1_w, eye1_h), (eye2_x, eye2_y, eye2_w, eye2_h) = eyes[:2]
            
            # Calculate the center of each eye
            eye1_center = (eye1_x + eye1_w // 2, eye1_y + eye1_h // 2)
            eye2_center = (eye2_x + eye2_w // 2, eye2_y + eye2_h // 2)
            
            # Calculate the horizontal distance between the centers
            eye_distance = abs(eye1_center[0] - eye2_center[0])
            
            return eye_distance, x, y, w, h  # Return a tuple when eyes are detected
        
    return 0, x, y, w, h  # Return default value if no face is found

def smile_detection(image, face_cascade, smile_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    x, y, w, h = 0, 0, 0, 0  # Initialize these variables before the loop
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        
        # Smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)           
        return len(smiles), x, y, w, h
    
    return 0, x, y, w, h  # Return default value if no face is found, along with the last values of x, y, w, h


def detect_greeting_words(text):
  """Detects the greeting words "Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening", "How are you?", "How's it going?", "What's up?", "Nice to see you", "Long time no see", "It's good to see you again", "It's a pleasure to meet you", and "How can I help you?" in the text.

  Args:
    text: The text to search for greeting words in.

  Returns:
    A list of greeting words found in the text.
  """
  greeting_words_regex = re.compile(r'(?i)\b(hello|hi|hey|good morning|good afternoon|good evening|how are you|how\'s it going|what\'s up|nice to see you|long time no see|it\'s good to see you again|it\'s a pleasure to meet you|how can I help you|namaskar|namastey|pranam|sat shri akal)\b')


  greeting_words = []
  for match in greeting_words_regex.finditer(text):
    greeting_words.append(match.group())
  return greeting_words

# load trained model for emotion detection
emotion_model = load_model("best_model.h5")

def get_emotion_change(face_cascade,image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)  # convert to RGB
        roi_rgb = cv2.resize(roi_rgb, (224, 224))  # resize to (224, 224)
        img_pixels = np.expand_dims(roi_rgb, axis=0)
        img_pixels = img_pixels / 255.0  # Normalize pixel values to [0, 1]

        predictions = emotion_model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        return predicted_emotion
    return None


def detect_voice_pauses(audio_file_path):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)

    # Set the silence threshold for voice activity detection
    silence_threshold = -40  # Adjust this threshold based on your audio characteristics

    # Split the audio on silence to get segments with voice activity
    voice_segments = split_on_silence(audio, silence_thresh=silence_threshold)

    # Calculate total voice duration without pauses
    total_voice_duration_without_pauses = sum(len(segment) for segment in voice_segments)

    # Calculate total voice duration with pauses
    total_voice_duration_with_pauses = len(audio)

    # Check if total_voice_duration_with_pauses is zero before calculating the ratio
    if total_voice_duration_with_pauses == 0:
        # Handle the case where total_voice_duration_with_pauses is zero
        return "Unable to detect pauses"

    # Calculate the ratio of voice duration without pauses to total voice duration
    ratio = total_voice_duration_without_pauses / total_voice_duration_with_pauses

    if 0.5 < ratio < 0.9:
        return "Pauses seem natural"
    else:
        return "Pauses seem unnatural"



def voice_emotion(audio_file_path):
    # Load the model
    filename = 'modelForPrediction1.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # Extract features from the trimmed audio file
    new_feature = extract_feature(audio_file_path, mfcc=True, chroma=True, mel=True)

    if new_feature is not None:
        new_feature = new_feature.reshape(1, -1)
        prediction = loaded_model.predict(new_feature)
        return prediction
    else:
        print("Error extracting features for prediction.")



def analyze_language_and_voice(audio_file_path):
    # Transcribe spoken words
    transcribed_text = transcribe_audio(audio_file_path)
    frequently_used_words = get_frequently_used_words(transcribed_text)
    words_list = []
    for word, frequency in frequently_used_words:
        words_list.append(word)

    # Analyze language characteristics (e.g., sentiment)
    language_analysis = analyze_language(transcribed_text)

    #Detect Greeting in voice
    greeting_words = detect_greeting_words(transcribed_text)

    # Analyze voice energy level
    audio = AudioSegment.from_wav(audio_file_path)
    energy_level = calculate_energy_level(audio)
    # Categorize energy level
    energy_category = categorize_energy_level(energy_level)
    filler_words = analyze_filler_words(transcribed_text)
    # Analyze voice modulation
    voice_modulation = analyze_voice_modulation(audio_file_path)
    return language_analysis, voice_modulation,energy_category,filler_words,words_list,greeting_words

# Initialize MediaPipe Hands for hand movement detection
moving_hands = mp.solutions.hands
move = moving_hands.Hands()

# Initialize MediaPipe Hands for gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load your gesture recognition model and classNames
# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

def hand_movement(image):
    x = 0
    y = 0
    try:
        # Process the image with MediaPipe Hands
        hands_results = move.process(image)

        # Check if hands are detected
        if hands_results.multi_hand_landmarks:
            return ("Hand Moving", x, y)
    except Exception as e:
        print(f"Error in hand_movement: {e}")
    return None

def get_thanks_gesture(image):
    x = 0
    y = 0
    hands_results = hands.process(image)

    # Check if hands are detected
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Define your logic for detecting a "thanks" gesture
            if thumb_tip.y < index_tip.y and middle_tip.y < index_tip.y:
                return ('Thanks Gesture', x, y)

            # Draw hand landmarks on the image (optional for visualization)
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return None

    return None

def hand_greeting_gesture(frame):
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = move.process(framergb)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]

            return className

    return None



def body_confidence(image):
    pose_results = pose.process(image)
    if pose_results.pose_landmarks:
        # Access landmarks and analyze body posture
        # You can define your own rules for confident body posture
        # For simplicity, let's consider if shoulders are aligned
        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if left_shoulder.x < right_shoulder.x:
            posture = "Confident"
        else:
            posture = "Not Confident"
        return posture

def get_frequently_used_words(transcribed_text):
    if transcribed_text == "Transcription could not be performed":
        most_common_words = []
        return most_common_words
    # Tokenize the transcribed text into words
    words = word_tokenize(transcribed_text)
    
    # Calculate the frequency distribution of words
    freq_dist = FreqDist(words)
    # Get the most common words
    most_common_words = freq_dist.most_common(10)  # You can adjust the number based on your preference

    return most_common_words

def analyze_filler_words(transcribed_text):
    # Define a list of common filler words
    filler_words = ["um", "uh", "like", "you know", "so","very", "actually", "basically", "literally", "well", "uhm", "uhh", "okay", "right", "I mean", "sort of", "kind of", "definitely", "obviously", "seriously", "totally", "absolutely", "basically", "essentially", "apparently", "apparently", "frankly", "honestly", "clearly", "you see", "mind you", "anyway", "however", "meanwhile", "nevertheless", "otherwise", "somehow", "therefore", "anyhow", "consequently", "furthermore", "otherwise", "moreover"]

    # Convert text to lowercase for case-insensitive matching
    transcribed_text_lower = transcribed_text.lower()

    used_filler_words = list(set(word for word in filler_words if word in transcribed_text_lower))

    return used_filler_words

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

    try:
        transcribed_text = recognizer.recognize_google(audio_data)
        return transcribed_text
    except sr.UnknownValueError:
        return "Transcription could not be performed"

def analyze_language(text):
    # Initialize the SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Get sentiment scores
    sentiment_scores = sid.polarity_scores(text)
    sentiment_score_average = (sentiment_scores['compound'] + 1) / 2


    # Determine sentiment based on the compound score
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    language_data = {
        "sentiment": sentiment,
        "sentiment_scores": sentiment_scores,
        "sentiment_score_average": sentiment_score_average
    }
    return language_data

def voice_monotone(audio_file_path):
    audio = AudioSegment.from_file(audio_file_path)
    samples = np.array(audio.get_array_of_samples())
    # Calculate the root mean square (RMS) of the audio signal
    rms = np.sqrt(np.mean(np.square(samples)))

    # Print the RMS value for reference
    # print("RMS:", rms)

    # Set a threshold for determining monotone or clear voice
    min_threshold = 20  # Adjust this threshold based on your observations
    max_threshold = 50

    # Check if the RMS value is below the threshold
    if min_threshold < rms < max_threshold:
        return "Voice is clear."
    else:
        return "Voice is monotone."

def analyze_voice_modulation(audio_file_path):
    audio = AudioSegment.from_wav(audio_file_path)

    # Perform voice modulation analysis (add your logic here)

    pitch = audio.dBFS

    # Check if pitch is -inf and handle it
    if pitch == float('-inf'):
        pitch = None

    min_pitch = -40
    max_pitch = 0
    # Calculate the percentage of voice modulation
    if pitch is not None:
        percentage_modulation = (pitch - min_pitch) / (max_pitch - min_pitch) * 100
        percentage_modulation = max(0, min(100, percentage_modulation))  # Ensure it's within the 0-100 range
    else:
        percentage_modulation = 0.0

    # You can define your own criteria for rating voice modulation
    if pitch is not None:
        if pitch > -12:
            modulation_rating = "Excellent"
        elif -12 >= pitch >= -27:
            modulation_rating = "Good"
        else:
            modulation_rating = "Needs Improvement"
    else:
        modulation_rating = "Not Available"

    return {
        "pitch": round(pitch,2),  # Pitch in dB
        "percentage_modulation": round(percentage_modulation, 2),
        "modulation_rating": modulation_rating
    }
    

def calculate_energy_level(audio):
    # Calculate the root mean square (RMS) to estimate energy level
    rms = np.sqrt(np.mean(np.square(audio.get_array_of_samples())))
    
    # Adjust the range as needed
    min_rms = 0.0  # Set the minimum expected RMS value
    max_rms = 100.0  # Set the maximum expected RMS value
    
    # Normalize the RMS value to the range [0, 1]
    normalized_energy = (rms - min_rms) / (max_rms - min_rms)
    return normalized_energy

def categorize_energy_level(energy_level):
    # Define thresholds for categorization
    low_threshold = 0.3
    high_threshold = 0.7
    energy_score = round((energy_level * 100),2)
    if energy_level < low_threshold:
        return ("Low",energy_score)
    elif low_threshold <= energy_level < high_threshold:
        return ("Medium",energy_score)
    else:
        return ("High",energy_score)

def generate_audio_file(video_path):
    # Speech recognition setup
    recognizer = sr.Recognizer()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    audio_file_path = f"speeches/file_{timestamp}.wav"  # Use a timestamp to create a unique audio file name

    # Use ffmpeg to extract audio from the video
    ffmpeg_path = "/usr/bin/ffmpeg"  # Replace with the actual path
    video_to_audio_command = [ffmpeg_path, "-i", video_path, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", audio_file_path]

    try:
        subprocess.run(video_to_audio_command, check=True)
        print("Audio file generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    
    return audio_file_path

def calculate_speech_rate(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio_data)
        word_count = len(text.split())
    except sr.UnknownValueError:
        word_count = 0

    # Use the sample rate from AudioFile object
    frame_rate = source.SAMPLE_RATE

    # Assume speech duration is the length of the audio in seconds
    speech_duration = len(audio_data.frame_data) / frame_rate

    # Check if speech_duration is zero before calculating speech rate
    if speech_duration == 0:
        # Handle the case where speech_duration is zero (e.g., empty audio file)
        speech_rate = 0
    else:
        # Calculate speech rate in words per minute
        speech_rate = (word_count / speech_duration) * 60

    return speech_rate


def analized_video_detail(request,video_id):
    try:
        video_data = VideoRecognition.objects.get(id=video_id)
    except:
        video_data=None
    return render(request,"video_detail.html",{
        "video_data":video_data
    })

def analized_video_list(request):
    all_data = VideoRecognition.objects.all()
    return render(request,"analize_video_list.html",{
        "all_data":all_data
    })

def image_analysis(request):
    if request.method == "POST":
        img = request.FILES.get("image")
        print(img)
        try:
            # Check if an image with the same name exists
            image_data = ImageRecognition.objects.get(name=img)
            # If exists, return the existing data
            return render(request, "upload_image.html", {
                "message": "Image already exists with this name and this image data shown below.",
                "tag": "info",
                "image_data":image_data,
                "face_analysis": ast.literal_eval(image_data.dominant_emotion)
            })
        except ImageRecognition.DoesNotExist:
            # If not exists, perform face analysis and save the new image data
            if img:
                # Save the uploaded image to a temporary file
                with open("temp_image.jpg", "wb") as f:
                    for chunk in img.chunks():
                        f.write(chunk)

                # Restrict TensorFlow to use only CPU
                tf.config.set_visible_devices([], 'GPU')

                # Analyze image using DeepFace
                face_analysis = DeepFace.analyze(img_path="temp_image.jpg", enforce_detection=False, detector_backend='mtcnn')
                emotions = [data['dominant_emotion'] for data in face_analysis]

                # Save analysis results to the database
                image_recognition = ImageRecognition(name=img, image=img,image_analysis_data = face_analysis,dominant_emotion=str(emotions))
                image_recognition.save()
                image_data = ImageRecognition.objects.get(name=img)
                # You may want to delete the temporary file after analysis
                os.remove("temp_image.jpg")

                return render(request, "upload_image.html", {
                    "face_analysis": emotions,
                    "image_data":image_data,
                })
            else:
                pass

    return render(request, "upload_image.html")

def analized_image_list(request):
    all_data = ImageRecognition.objects.all()
    return render(request,"analize_image_list.html",{
        "all_data":all_data
    })

def analyze_pdf(request):
    if request.method == "POST":
        pdf_file = request.FILES.get("pdf")  # Use request.FILES to handle file uploads
        if pdf_file:
            try:
                # Check if PDF with the same name exists
                pdf_data = AnalyzePDF.objects.filter(pdf_name=pdf_file).first()
                print(pdf_data)

                if pdf_data is None:
                    # Convert the InMemoryUploadedFile to a BytesIO object
                    pdf_content = pdf_file.read()
                    
                    # Extract text using pdfminer
                    all_text = extract_text(BytesIO(pdf_content))
                    print(all_text)

                    # Save to the database
                    pdf_data = AnalyzePDF(pdf_name=pdf_file, pdf_file=pdf_file, pdf_text=all_text)
                    pdf_data.save()

                    return render(request, "pdf_upload.html", {
                        "pdf_name": pdf_file,
                        "all_text": all_text,
                    })
                else:
                    return render(request, "pdf_upload.html", {
                        "message": "PDF already exists with the same name.",
                        "tag": "danger",
                    })
            except Exception as e:
                print(f"Error reading PDF: {e}")
                return render(request, "pdf_upload.html", {
                    "message": "Something went wrong. Please try again.",
                    "tag": "danger",
                })
        else:
            print("No PDF file uploaded")
            return render(request, "pdf_upload.html", {
                "message": "No PDF file uploaded",
                "tag": "danger",
            })

    return render(request, "pdf_upload.html")


def analyzed_pdf_list(request):
    all_pdf = AnalyzePDF.objects.all()
    return render(request,"analized_pdf_list.html",{
        "all_pdf":all_pdf
    })

def analyzed_pdf_view(request,pdf_id):
    pdf = AnalyzePDF.objects.get(id = pdf_id)
    return render(request,"analyzed_pdf_view.html",{
        "pdf":pdf
    })


class VideoUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    serializer_class = VideoSerializer
    
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        print(serializer)
        if serializer.is_valid():
            try:
                user = serializer.validated_data['user']
                print(user)
            except KeyError:
                error_response = {
                    "message": "User ID field is required.",
                }
                return Response(error_response, status=status.HTTP_400_BAD_REQUEST)
            try:
                video_file = serializer.validated_data['video_file']
            except KeyError:
                error_response = {
                    "message": "Video field is required. Please check video field correctly defined.",
                }
                return Response(error_response, status=status.HTTP_400_BAD_REQUEST)
                 
            
            if video_file:
                if video_file.size > 10 * 1024 * 1024:
                    message = "Video size should be less than 10MB."
                    return Response({"message": message}, status=status.HTTP_400_BAD_REQUEST)
                # Create a temporary file to store the uploaded video
                temp_video_path = f"{video_file.name}"
                with open(temp_video_path, 'wb') as temp_file:
                    for chunk in video_file.chunks():
                        temp_file.write(chunk)

                # Use moviepy to get the video duration
                clip = VideoFileClip(temp_video_path)
                duration = clip.duration

                # Check if video duration is greater than 30 seconds
                if duration > 30:
                    os.remove(f"{video_file}")
                    message = "Video duration should be less than 30 seconds."
                    return Response({"message": message}, status=status.HTTP_400_BAD_REQUEST)
                
                try:
                    video_data = VideoRecognition.objects.get(name=f"{video_file}")
                except VideoRecognition.DoesNotExist:
                    video_data = None

                if video_data is None:
                    analysis = analyse_video(video_file,user,duration)
                    if analysis is not None:
                        analysed_data = VideoRecognition.objects.get(id = analysis)
                        serializer = VideoDataSerializer(analysed_data)  # Use your VideoDataSerializer to serialize the instance
                        serialized_data = serializer.data
                        return Response(
                            serialized_data,
                            status=status.HTTP_200_OK
                        )
                    else:
                        analysed_data = {
                            "message":"Error during video analysis. Please try again or provide a different video."
                        }
                        return Response(
                            analysed_data,
                            status=status.HTTP_400_BAD_REQUEST,
                            content_type="application/json" 
                        )
                else:
                    analysed_data = {
                        "message":"Video already with this name."
                    }
                    os.remove(temp_video_path)
                    return Response(
                        analysed_data,
                        status=status.HTTP_409_CONFLICT,
                        content_type="application/json"  # Set content type to application/json
                    )
            else:
                analysed_data = {
                    "message":"Video not found,Please upload video."
                }
                return Response(
                    analysed_data,
                    status=status.HTTP_404_NOT_FOUND,
                    content_type="application/json"  # Set content type to application/json
                )
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
    
def analyse_video(video_file,user,duration):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')   
    try:
        audio_file_path = generate_audio_file(f"{video_file}")
        print(audio_file_path)
        language_sentiment_analysis, voice_modulation_data,energy_category,filler_words,words_list,greeting_words = analyze_language_and_voice(audio_file_path)
        # Get speech rate
        wpm = calculate_speech_rate(audio_file_path)
        speech_rate = round(wpm,2)
        monotone = voice_monotone(audio_file_path)
        pauses = detect_voice_pauses(audio_file_path)
        language_analysis = language_sentiment_analysis["sentiment"]
        
        energy_level,energy_score = energy_category
        voice_modulation = {"pitch":voice_modulation_data["pitch"],"modulation_rating":voice_modulation_data["modulation_rating"]}
    
        if len(greeting_words) > 0:
            greeting = "Greeting included"
        else:
            greeting = None
        emo = voice_emotion(audio_file_path)
        # Convert NumPy array to Python list
        voice_emo = emo.tolist() if isinstance(emo, np.ndarray) else emo
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    cap = cv2.VideoCapture(f"{video_file}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Open a video capture object
    output_filename = f"output_{video_file}"
    video_output = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
    # Variables 
    blink_thresh = 0.45
    succ_frame = 2
    count_frame = 0
    blinks_per_minute = 0
    current_second_start_time = time.time()
    # Example: Process every 11rd frame
    frame_skip = 11
    frame_count = 0  # Initialize the frame count
    eye_contact = None
    hand_move = None
    eye_bling = None
    b_confidence = None
    thanks = None
    greet_gesture = None
    # Initialize variables for tracking time
    start_time = time.time()
    total_detected_time = 0
    total_not_detected_time = 0
    # Initialize variables for Body Posture
    good_posture_time = 0
    bad_posture_time = 0   
    hand_movement_count = 0
    none_hand_movement_count = 0
    emotion_change = 0
    emotion_not_detected = 0
    eye_contact_detect = 0
    eye_not_contact = 0 
    body_confidence_count = 0
    not_body_confidence_count = 0

    # List of possible emotions
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    # Initialize a dictionary to store emotion counts
    emotion_counts = {emotion: 0 for emotion in emotions}

    # Create a VideoRecognition object and associate it with the video_frame
    video_recognition = VideoRecognition.objects.create(name=str(video_file))

    while True:
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null Frames")
            break

        # Specify the output thumbnail filename
        file = str(video_file)
        thumb = file.replace(".mp4", "")
        thumbnail_filename = f"thumbnail_{thumb}.jpg"

        # Save the first frame as a thumbnail image
        cv2.imwrite(thumbnail_filename, image)

        current_time = frame_count / fps
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames
        try:                   
            posture = detect_body_posture(image, fps)
            good_time, bad_time = posture
            print("Good Posture Time:", good_time)
            print("Bad Posture Time:", bad_time)
            if good_time > 0:
                save_detected_frame(video_recognition, "good_posture", image,frame_count,current_time)
                good_posture_time += good_time

            else:
                bad_posture_time += bad_time
                save_detected_frame(video_recognition, "bad_posture", image,frame_count,current_time)

        except Exception as e:
            print(e)
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = detector(gray_frame)
        if len(faces) > 0:
            total_detected_time += time.time() - start_time
            start_time = time.time()
            save_detected_frame(video_recognition, "face_detected", image,frame_count,current_time)
            for face in faces:
                # Get facial landmarks
                landmarks = landmark_predict(gray_frame, face)
                # Draw circles around each landmark point
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                # Draw rectangle around the face
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            total_not_detected_time += time.time() - start_time
            start_time = time.time()     
            save_detected_frame(video_recognition, "face_not_detected", image,frame_count,current_time)           
        frame = cv2.flip(image, 1)
        # Emotion Changes Detection
        predicted_emotion = get_emotion_change(face_cascade,image)
        # Update the count for the detected emotion
        if predicted_emotion in emotion_counts:
            emotion_counts[predicted_emotion] += 1
        else:
            pass

        if predicted_emotion is not None:
            emotion_change += 1
            cv2.putText(image, predicted_emotion, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            save_detected_frame(video_recognition, f"{predicted_emotion}", image,frame_count,current_time)
        else:
            emotion_not_detected += 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)               
        # Process the image with MediaPipe Hands
        # Hand Movement, Thanks Geesture and body confidence detection code functions *****************
        greeting_gesture = hand_greeting_gesture(frame)
        hand_track = hand_movement(image)
        thanks_gesture = get_thanks_gesture(image)
        confidence = body_confidence(image)
        # Convert the RGB image to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if greeting_gesture == "Namaste" or greeting_gesture == "Hi/Hello":
            greet_gesture = "Greeting gesture included"
            cv2.putText(image, greeting_gesture, (20, 100), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
            save_detected_frame(video_recognition, "greeting_gesture", image,frame_count,current_time)
        else:
            save_detected_frame(video_recognition, "no_gesture", image,frame_count,current_time)
        if hand_track is not None:
            hand_move = 'Hand Moving'
            hand_movement_count += 1
            hand_track,x,y = hand_track
            cv2.putText(image, 'Hand Moving', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            save_detected_frame(video_recognition, "hand_moving", image,frame_count,current_time)
        else:
            none_hand_movement_count += 1
            save_detected_frame(video_recognition, "hand_not_moving", image,frame_count,current_time)

        if thanks_gesture is not None:
            thanks_gesture,x,y = thanks_gesture
            thanks = "Thanking gesture included"
            cv2.putText(image, 'Thanks Gesture', (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            save_detected_frame(video_recognition, "thanks", image)
        else:
            save_detected_frame(video_recognition, "no_thanks", image,frame_count,current_time)
        if confidence == "Confident":
            b_confidence = "Confident body posture"
            body_confidence_count += 1
            # Display the posture on the frame
            cv2.putText(image, f"Posture: {confidence}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 51 ), 2)
            save_detected_frame(video_recognition, "confident", image,frame_count,current_time)
        else:
            not_body_confidence_count += 1
            cv2.putText(image, f"Posture: {confidence}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 153, 51 ), 2)
            save_detected_frame(video_recognition, "unconfident", image,frame_count,current_time)
        # Eye Contact detection code start ***********
        eye_distance, x, y, w, h = eye_contact_detection(image, face_cascade, eye_cascade)
        eye_contact_threshold = 20  # Example threshold, you may need 
        # Check if eyes are horizontally aligned (within the threshold)
        if eye_distance < eye_contact_threshold:
            eye_contact = 'Eye Contact'
            eye_contact_detect += 1
            cv2.putText(image, 'Eye Contact', (x, y + h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 233, 51), 2) 
            save_detected_frame(video_recognition, "eye_contact", image,frame_count,current_time)
        else:
            eye_not_contact += 1     
            save_detected_frame(video_recognition, "not_contact", image,frame_count,current_time)          
        # Eye Blinging detection code start *************
        blinging_detected = eye_blinging(image)
        if blinging_detected < blink_thresh: 
            count_frame += 1  # incrementing the frame count 
        else: 
            if count_frame >= succ_frame: 
                blinks_per_minute += 1
                cv2.putText(image, 'Blink Detected', (60, 50), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1) 
            else: 
                count_frame = 0
        elapsed_time = time.time() - current_second_start_time
        if elapsed_time >= 5:
            if blinks_per_minute > 2:
                eye_bling = "Blink more often"
                cv2.putText(image, f'{blinks_per_minute} Blinks in 5 Second', (60, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                save_detected_frame(video_recognition, "more_blinging", image,frame_count,current_time)
            blinks_per_minute = 0
            current_second_start_time = time.time()
        # Eye Blinging detection code start *************
        smile_detect, x, y, w, h = smile_detection(image, face_cascade, smile_cascade)
        if smile_detect > 0:
            cv2.putText(image, 'Smiling', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)               
        # Write the frame to the output video.
        video_output.write(image)
        # new_width = 500
        # new_height = 600
        # # Resize the image
        # resized_image = cv2.resize(image, (new_width, new_height))
        # # Display the frame.
        # cv2.imshow('Video', resized_image)
        # # Break the loop if 'q' key is pressed.
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break                
    # Release video capture and writer objects.
    cap.release()
    video_output.release()
    cv2.destroyAllWindows()

    # Find the emotion with the maximum count
    most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)

    # Analysis score detection code start ##################
    if good_posture_time > 0:
        posture_ratio = good_posture_time/(good_posture_time + bad_posture_time)
        if posture_ratio > 0.5:
            body_posture = "Good Body Posture"
        else:
            body_posture = "Bad Body Posture"
    else:
        posture_ratio = 0.0
        body_posture = None
    if hand_movement_count > 0:
        hand_move_ratio = hand_movement_count / (hand_movement_count + none_hand_movement_count) 
    else:
        hand_move_ratio = 0.0
    body_language_ratio = ((posture_ratio + hand_move_ratio)/2) * 100
    body_language_score = round(body_language_ratio,2)
    if emotion_change > 0:
        emotion_change_ratio = (emotion_change/(emotion_change + emotion_not_detected)) * 100
    else:
        emotion_change_ratio = 0.0
    if eye_contact_detect > 0:
        eye_contact_ratio = (eye_contact_detect/(eye_contact_detect + eye_not_contact)) * 100
    else:
        eye_contact_ratio = 0.0
    facial_expression_ratio = (emotion_change_ratio + eye_contact_ratio)/2
    facial_expression_score = round(facial_expression_ratio,2)

    if not filler_words:
        language_analysis_score = language_sentiment_analysis["sentiment_score_average"] * 100
        language_analysis_average = 0.0
    else:
        language_analysis_average = ((language_sentiment_analysis["sentiment_score_average"] + 1.0) / 2) * 100  # 1.0 is added for filler words average to get percentage
        language_analysis_score = round(language_analysis_average, 2)

    if body_confidence_count > 0:
        confidence_ratio = body_confidence_count/(body_confidence_count + not_body_confidence_count)
    else:
        confidence_ratio = 0.0
    
    body_confidence_score = round((confidence_ratio * 100),2)

    voice_modulation_score = round((energy_score + voice_modulation_data["percentage_modulation"])/2,2)

    total_len = total_detected_time + total_not_detected_time
    ratio = total_detected_time/total_len
    if ratio > 0.5:
        face_detected = "Appropriate Facial Detected."
    else:
        face_detected = "Appropriate Facial Not Detected." 

    # Getting Total Video Analysis Score ####################
    t_score = get_analysis_score(body_language_score,facial_expression_score,voice_modulation_score,body_confidence_score,language_analysis_score)   
    try:
        # Update the fields with new data
        video_recognition.thumb_img = File(open(thumbnail_filename, 'rb'))
        video_recognition.analysis_score = t_score
        video_recognition.language_analysis = language_analysis
        video_recognition.voice_modulation_analysis = voice_modulation
        video_recognition.energy_level_analysis = energy_level
        video_recognition.video_file = video_file
        video_recognition.video_durations = duration
        video_recognition.word_per_minute = speech_rate
        video_recognition.filler_words_used = filler_words
        video_recognition.frequently_used_word = words_list
        video_recognition.voice_emotion = voice_emo
        video_recognition.confidence = b_confidence
        video_recognition.eye_bling = eye_bling
        video_recognition.hand_movement = hand_move
        video_recognition.eye_contact = eye_contact
        video_recognition.thanks_gesture = thanks
        video_recognition.greeting = greeting
        video_recognition.greeting_gesture = greet_gesture
        video_recognition.voice_tone = monotone
        video_recognition.voice_pauses = pauses
        video_recognition.appropriate_facial = face_detected
        video_recognition.body_posture = body_posture
        video_recognition.body_language_score = body_language_score
        video_recognition.facial_expression_score = facial_expression_score
        video_recognition.language_analysis_score = language_analysis_score
        video_recognition.voice_modulation_score = voice_modulation_score
        video_recognition.body_confidence_score = body_confidence_score
        video_recognition.facial_expression = most_frequent_emotion
        # Save the updated object
        video_recognition.save() 
        data = video_recognition.id              
    except Exception as e:
        data = None       
    try:
        os.remove(f"{video_file}")
        os.remove(f"output_{video_file}")
        os.remove(audio_file_path)
        os.remove(thumbnail_filename)
    except:
        pass
    return data
        

# Analysed video list api code **********************************
class AnalysedVideoListView(APIView):
    def get(self, request):
        user_id = request.query_params.get('user', None)
        if user_id is not None:
            # Filter data based on the provided user_id
            filtered_data = VideoRecognition.objects.filter(user=user_id)
            serializer = VideoDataListSerializer(filtered_data, many=True)
            return Response(
                {
                    "message":"Data found",
                    "data":serializer.data,
                },
                status=status.HTTP_200_OK
            )
        else:
           return Response(
                {
                    "message":"User ID not found",
                },
                status=status.HTTP_404_NOT_FOUND
            )


#Image analysis API code start ******************************************

class ImageAnalysisView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    serializer_class = ImageSerializer
    
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            try:
                image_file = serializer.validated_data['image']
            except KeyError:
                error_response = {
                    "message": "Image field is required. Please check image field correctly defined.",
                }
                return Response(error_response, status=status.HTTP_400_BAD_REQUEST)      
            
            try:
                image_data = ImageRecognition.objects.get(name=image_file)
                image_data = {
                    "message":"Image already with this name."
                }
                return Response(
                    image_data,
                    status=status.HTTP_409_CONFLICT,
                    content_type="application/json"  # Set content type to application/json
                )
            except ImageRecognition.DoesNotExist:
                if image_file:
                    # Check image size
                    if hasattr(image_file, 'size') and image_file.size > 10 * 1024 * 1024:
                        message = "Image size should be less than 10MB."
                        return Response({"message": message}, status=status.HTTP_400_BAD_REQUEST)
                    
                    # Save the uploaded image to a temporary file
                    with open("temp_image.jpg", "wb") as f:
                        for chunk in image_file.chunks():
                            f.write(chunk)

                    # Restrict TensorFlow to use only CPU
                    tf.config.set_visible_devices([], 'GPU')

                    # Analyze image using DeepFace
                    face_analysis = DeepFace.analyze(img_path="temp_image.jpg", enforce_detection=False, detector_backend='mtcnn')
                    emotions = [data['dominant_emotion'] for data in face_analysis]

                    # Save analysis results to the database
                    image_recognition = ImageRecognition(name=image_file, image=image_file,image_analysis_data = face_analysis,dominant_emotion=str(emotions))
                    image_recognition.save()
                    image_data = ImageRecognition.objects.get(name=image_file)
                    # You may want to delete the temporary file after analysis
                    os.remove("temp_image.jpg")

                    serializer = ImageDataSerializer(image_data)  # Use your VideoDataSerializer to serialize the instance
                    serialized_data = serializer.data
                    return Response(
                        serialized_data,
                        status=status.HTTP_200_OK
                    )
                else:
                    image_data = {
                        "message":"Image not found,Please upload an image."
                    }
                    return Response(
                        image_data,
                        status=status.HTTP_404_NOT_FOUND,
                        content_type="application/json" 
                    )
        
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )


# Analysed video list api code **************************
class AnalysedImageListView(APIView):
    def get(self,request):
        all_data = ImageRecognition.objects.all()
        serializer = ImageDataListSerializer(all_data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

# Analysed video list api code **********************************
class AnalysedVideoDetailView(APIView):
    def get(self, request):
        video_id = request.query_params.get('id', None)
        if video_id is not None:
            # Filter data based on the provided video_id
            video_data = VideoRecognition.objects.get(id=video_id)
            serializer = VideoDataListSerializer(video_data, many=True)
            return Response(
                {
                    "message":"Data found",
                    "data":serializer.data,
                },
                status=status.HTTP_200_OK
            )
        else:
           return Response(
                {
                    "message":"User ID not found",
                },
                status=status.HTTP_404_NOT_FOUND
           )
            
def delete_data(request):
    # Delete all instances of Frame model
    Frame.objects.all().delete()

    # Delete all instances of Posture model
    Posture.objects.all().delete()

    # Delete all instances of DetectedFrames model
    DetectedFrames.objects.all().delete()

    # Delete all instances of VideoRecognition model
    VideoRecognition.objects.all().delete()

    return HttpResponse("Success")

def video_detail(request,video_id):
    video = VideoRecognition.objects.get(id=video_id)
    if request.method == "POST":
        posture_name = request.POST.get("posture")
        posture = Posture.objects.filter(video=video,name = posture_name)

    return render(request,"detail.html",{
        "video_data":video
    })

def frame(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            video_id = data.get("id")
            posture_name = data.get("posture")
            
            video = VideoRecognition.objects.get(id=video_id)
            posture = Posture.objects.get(video=video, name=posture_name)
            detected_frames = DetectedFrames.objects.get(posture=posture)

            # Construct the redirect URL dynamically
            redirect_url = reverse('get_data', kwargs={'id': detected_frames.id})

            # Return JSON response with the data
            return JsonResponse({
                "redirect_url": redirect_url
            })
        except Exception as e:
            print(str(e))  # Print the error for debugging
            return JsonResponse({"error": str(e)}, status=500)

    return render(request, "detail.html")

def get_data(request,id):
    detected_frames = DetectedFrames.objects.get(id=id)
    video = detected_frames.posture.video
    video_duration = detected_frames.posture.video.video_durations
    posture = detected_frames.posture
    frames = detected_frames.frames.all()
    total_frame = len(frames)

    # Calculate the percentage position for each frame
    for frame in frames:
        if frame.current_time and video_duration is not None:
            frame.percent_position = frame.current_time / video_duration * 100
        else:
            frame.percent_position = 0.0

    return render(request,"frame_data.html",{
        'frame_data': frames,
        'total_frame': total_frame,
        "video":video,
        "posture":posture
    })


# Analysed video list api code **********************************
class VideoFrameView(APIView):
    def get(self, request):
        video_id = request.query_params.get('id', None)
        posture_name = request.query_params.get('posture', None)
        if video_id and posture_name is not None:
            video = VideoRecognition.objects.get(id=video_id)
            posture = Posture.objects.get(video=video, name=posture_name)
            detected_frames = DetectedFrames.objects.get(posture=posture)
            serializer = VideoFrameSerializer(detected_frames)
            return Response(
                {
                    "message":"Data found",
                    "data":serializer.data,
                },
                status=status.HTTP_200_OK
            )
        else:
           return Response(
                {
                    "message":"User ID not found",
                },
                status=status.HTTP_404_NOT_FOUND
           )
            