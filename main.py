import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import cvzone
import math
import pickle
import firebase_admin
from firebase_admin import storage, credentials, db
from datetime import datetime

cred = credentials.Certificate("C:/Users/Admin/ML_MINIPROJECT/serviceAccountKey.json")

try:
    app = firebase_admin.get_app()
except ValueError:
    app = firebase_admin.initialize_app(cred, {
        'databaseURL': "https://ml-miniproject-6969-default-rtdb.firebaseio.com/",
        'storageBucket': "ml-miniproject-6969.appspot.com"
    })
bucket = storage.bucket(app=firebase_admin.get_app(), name='ml-miniproject-6969.appspot.com')

    

model = YOLO("best.pt")

conf_threshold = 0.5 

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

print("loading encoded file...")
file= open("encodefile.p",'rb')
encodelistknownwithid = pickle.load(file)
file.close()
print("loading complete")
encodelistknown,emp_id = encodelistknownwithid 
#print(emp_id)

emp_info=[]
imgemp=[]
for id in emp_id:
    emp_info.append(db.reference(f'emp/{id}').get())
    blob = bucket.get_blob(f'Images/{id}.png')
    arr = np.frombuffer(blob.download_as_string(),np.uint8)
    imgemp.append(cv2.imdecode(arr,cv2.COLOR_BGRA2BGR))
#print(imgemp)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
imageBackground = cv2.imread('Resources/AI - Powered Worker Safety (ml_miniproject).png')

counter = 0
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    person_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if classNames[cls] == "Person":
                x1, y1, x2, y2 = box.xyxy[0]
                person_boxes.append([x1, y1, x2, y2])
                break
    for person_box in person_boxes:
        person_img = img[int(person_box[1]):int(person_box[3]), int(person_box[0]):int(person_box[2])]
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                # cvzone.cornerRect(img, (x1, y1, w, h))
    
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                print(currentClass)
                if conf>0.5:
                    if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                        myColor = (0, 0,255)
                        counter+=1
                    elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                        myColor =(0,255,0)
                    else:
                        myColor = (255, 0, 0)
    
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                       colorT=(255,255,255),colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                if counter >=1 :
                    counter = 0
                    person_imgs = cv2.resize(person_img,(0,0),None,0.25,0.25)   
                    person_imgs = cv2.cvtColor(person_imgs,cv2.COLOR_BGR2RGB)
                    
                    facecurframe = face_recognition.face_locations(person_imgs)
                    encodecurframe = face_recognition.face_encodings(person_imgs,facecurframe)
                    
                    for encodeface,faceloc in zip(encodecurframe,facecurframe):
                        matches = face_recognition.compare_faces(encodelistknown, encodeface)
                        facedis = face_recognition.face_distance(encodelistknown, encodeface)
                        #print("matches",matches)
                        #print("facedis",facedis)
                        matchindex = np.argmin(facedis)
                        #print("matchindex",matchindex)
                        if matches[matchindex]:
                            print("known face detectecd with no ppe...")
                            print("employee_id", emp_id[matchindex])
                            y1,x2,y2,x1 = faceloc
                            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                            cv2.rectangle(person_img, (x1, y1), (x2, y2), (203,108,230), 3)   
                            datetimeobject = datetime.strptime(emp_info[matchindex]['last_violation'],"%Y-%m-%d %H:%M:%S")
                            secondelapsed = (datetime.now()-datetimeobject).total_seconds()
                            if secondelapsed>10:
                                ref = db.reference(f'emp/{emp_id[matchindex]}')
                                emp_info[matchindex]['safety_violation'] += 1
                                emp_info[matchindex]['last_violation'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                ref.child('safety_violation').set(emp_info[matchindex]['safety_violation'])
                                ref.child('last_violation').set(emp_info[matchindex]['last_violation'] )
                        else:
                            print("intuder alert...")
            
    #cv2.imshow("Image", img)
    enter = 0
    frame_resized = cv2.resize(img, (440, 341))
    for i in range(len(imgemp)):
        cv2.rectangle(imageBackground, (750, 145+enter), (1000, 175+enter), (255, 255, 255), -1)
        cv2.putText(imageBackground,"safety_violation : "+str(emp_info[i]['safety_violation']),(750,165+enter),
                    4,0.5,(50,50,50),1)
        #(w,h),_= cv2.getTextSize(emp_info[i]['name'],4,1,1)
        #offset = (250-w)//2
        cv2.putText(imageBackground,str(emp_info[i]['name']),(750,100+enter),
                    3,0.5,(50,50,50),1)
        cv2.putText(imageBackground,"department : "+str(emp_info[i]['department']),(750,125+enter),
                    4,0.5,(50,50,50),1)
        #cv2.putText(imageBackground,str(id),(1006,493),
        #            4,0.5,(255,255,255),1)
        cv2.putText(imageBackground,"last violation:",(750,195+enter),
                    4,0.5,(50,50,50),1)
        cv2.rectangle(imageBackground, (750, 200+enter), (1000, 230+enter), (255, 255, 255), -1)
        cv2.putText(imageBackground,str(emp_info[i]['last_violation']),(750,220+enter),
                    4,0.5,(50,50,50),1)
        imgemp_resized = cv2.resize(imgemp[i], (150, 150))
        imageBackground[80+enter:80+enter+150,584:584+150] = imgemp_resized
        enter = (i+1)*(200)
    imageBackground[303:303+341,62:62+440] = frame_resized
    cv2.imshow("database alert", imageBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()