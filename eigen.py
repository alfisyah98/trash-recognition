from time import ctime
import numpy, os
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import cv2
import pickle
from collections import Counter
import ntplib
  
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

pickle_in = open("pca1.dat","rb")
pca = pickle.load(pickle_in)

#X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)

pickle_in = open("clf1.dat","rb")
clf= pickle.load(pickle_in)


nama = {0: 'anorganik', 1: 'organik'}

#proses face recogn
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
sampah = cv2.CascadeClassifier('sampah1.xml')

cap.set(3,640)
cap.set(4,480)

identitas=[]
ketemu = False
#360x440
while(True):
    # Capture frame-by-frame
    test = []
    face = []
    ret, frame = cap.read()
    xv, yv, cv = frame.shape
    if ret == True :
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        trash = sampah.detectMultiScale(gray, 6.5, 17)
        test = []
        for (x,y,wf,hf) in trash:
            # create box on detected face
            frame = cv2.rectangle(frame,(x,y),(x+wf,y+hf),(255,0,0),1)
            
            wajah = frame[y:y+hf,x:x+wf]
            dim = (320, 320)
  
            # resize image
            resized = cv2.resize(wajah, dim, interpolation = cv2.INTER_AREA)
            resized=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            testImageFeatureVector=numpy.array(resized).flatten()
            test.append(testImageFeatureVector)
            testImagePCA = pca.transform(test)
            testImagePredict=clf.predict(testImagePCA)
            
            cv2.imshow('wframe', resized)
            print('ada wajah', nama[testImagePredict[0]], type(testImagePredict))
            cv2.putText(frame, "Name : " + nama[testImagePredict[0]], (x + x//10, y+hf+20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            #print('.',end='')
            
            identitas.append(testImagePredict[0])
            if len(identitas) >= 40 :
                ketemu = True
                break
            
        cv2.imshow('frame', frame)
        
        if ketemu == True :
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#print (identitas)
id_ = most_frequent(identitas)
print('ada wajah', nama[id_])

#def log(id):
    #openfile(detik)
    #simpan
def print_time():
    ntp_client = ntplib.NTPClient()
    response = ntp_client.request('asia.pool.ntp.org')
    print(ctime(response.tx_time))
print_time()
#if (id== 0,1,2,3):
    #buka_kunci()
    #log(id_)
