import glob
import cv2

sampah_cascade = cv2.CascadeClassifier('sampah.xml')

tujuan = glob.glob('./sampel foto/organik/*.jpg')

count = 0
for lok_gambar in tujuan:
    ini = cv2.imread(lok_gambar)
    gray = cv2.cvtColor(ini, cv2.COLOR_BGR2GRAY)
        
    trash = sampah_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,wf,hf) in trash:
        # create box on detected face
        roi = ini[y:y+hf, x:x+wf]
        cv2.imwrite('./data1/organik/'+str(count)+'.jpg', roi)
    count += 1            
    
print(tujuan)
