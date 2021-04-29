# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:24:27 2020

@author: vyaso
"""

import cv2
import face_recognition
webcam_video_stream = cv2.VideoCapture(0)


modi_image = face_recognition.load_image_file('C:/py/code/images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('C:/py/code/images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

ojesh_image = face_recognition.load_image_file('C:/py/code/images/samples/ojesh.jpg')
ojesh_face_encodings = face_recognition.face_encodings(ojesh_image)[0]


known_face_encodings = [modi_face_encodings, trump_face_encodings, ojesh_face_encodings]
known_face_names = ["Narendra Modi", "Donald Trump","Ojesh Vyas" ]

all_face_locations = []
all_face_encodings = []
all_face_names = []

while True:
    
    ret,current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
   
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
    
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)
    all_face_names = []
    
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
       
        name_of_person = 'Unknown face'
        
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
        
    cv2.imshow("Faces Identified",current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam_video_stream.release()
cv2.destroyAllWindows


