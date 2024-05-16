import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import storage, credentials

cred = credentials.Certificate("C:/Users/Admin/ML_MINIPROJECT/serviceAccountKey.json")

try:
    app = firebase_admin.get_app()
except ValueError:
    app = firebase_admin.initialize_app(cred, {
        'databaseURL': "https://ml-miniproject-6969-default-rtdb.firebaseio.com/",
        'storageBucket': "ml-miniproject-6969.appspot.com"
    })

folderpath = 'Images'
imgpathlist = os.listdir(folderpath)
#print(imgpathlist)
imglist=[]
emp_id =[]
for path in imgpathlist:
    imglist.append(cv2.imread(os.path.join(folderpath, path)))
    emp_id.append(os.path.splitext(path)[0])
    #print(path)
    #print(os.path.splitext(path)[0])
    filename = f'{folderpath}/{path}'
    bucket = storage.bucket(app=firebase_admin.get_app(), name='ml-miniproject-6969.appspot.com')
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)
#print(len(imglist))
#print(emp_id)

def find_encodings(imglist):
    encodelist = []
    for img in imglist:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
    
print("encoding started....")
encodelistknown = find_encodings(imglist)
encodelistknownwithid = [encodelistknown,emp_id]
#print("\n\n\n",encodelistknown,"\n\n\n")
print("encoding completed")

file= open("encodefile.p",'wb')
pickle.dump(encodelistknownwithid,file)
file.close()
print("file saved")