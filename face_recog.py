import cv2
import numpy as np
import os

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():           #if faces returns no value, return the same image screen available
        return img, []        #return image and empty list since we have no values of x,y,w,h2
    
    #creating a rectangle on the border of the face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi           #return image and the cropped photograph


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame) #function called
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = facemodel.predict(face) #prediction using the cropped image
        print(results)
        # harry_model.predict(face)
        
        #if results[1] < 500:
        confidence = int( 100 * (1 - (results[1])/400) )                #formula to find confidence score
        display_string = str(confidence) + '% Confident it is User'     #variable declared to store confidence percentage
        print(confidence)
        
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 80:
            cv2.putText(image, "Hey Yash", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            from twilio.rest import Client 
 
            account_sid = 'Account SID' 
            auth_token = 'Authentiation Token' 
            client = Client(account_sid, auth_token) 
 
            message = client.messages.create( 
                              from_='whatsapp:+1234567890',  
                              body='hi',      
                                to='whatsapp:+1234567890' 
                                            )
            print(message.sid)
            break
 
            
            # os.system("chrome https://www.google.com/)
            # os.system("wmplayer   c:\lw.mp3")
            #break
         
        else:
            
            cv2.putText(image, "I dont know, how r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()     
