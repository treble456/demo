import tkinter as tk
import os,dlib,glob,numpy
from skimage import io
import cv2
from PIL import Image,ImageTk
import imutils
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter.messagebox

import mysql.connector
company = mysql.connector.connect(
  host = "127.0.0.1",
  user = "root",
  password = "123456",
  database = "company",
  )
cursor=company.cursor()
root=tk.Tk()   #創建對象
root.title('Mall Face Recog')     #窗口的名稱
root.geometry('800x600') 



l=tk.Label(root,text='pictures will show in this place')   

l2=tk.Label(root,text='pictures will show in this place',font=('microsoft yahei', 16, 'bold'))

l3=tk.Label(root,text='pictures will show in this place')
l4=tk.Label(root,text='id')
l5=tk.Label(root,text='Name')
l6=tk.Label(root,text='telephone')
l7=tk.Label(root,text='likea')
l8=tk.Label(root,text='max')
object_text=tk.Entry()
object_text1=tk.Entry()
object_text2=tk.Entry()
object_text3=tk.Entry()
object_text4=tk.Entry()



predictor_path = "shape_68.dat"

face_rec_model_path = "dlib_f.dat"
def openpicture():
    filename=filedialog.askopenfilename()     
    filename1=str(filename)

    
    l.config(text=str(filename))

    


    img_path = '%s' %filename1
    
    
    faces_folder_path = "C:/Users/user/Downloads/Project/rec"
    
    image = cv2.imread(filename=img_path) 
    

    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    descriptors = []


    candidate = []
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        base = os.path.basename(f)

        candidate.append(os.path.splitext(base)[ 0])
        img = io.imread(f)


        dets = detector(img, 1)
        for k, d in enumerate(dets):

            shape = sp(img, d)

            face_descriptor = facerec.compute_face_descriptor(img, shape)

            v = numpy.array(face_descriptor)
            descriptors.append(v)
    img = io.imread(img_path)
    
    dets = detector(img, 1)
    
    dist = []
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = numpy.array(face_descriptor)

        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

        cv2.rectangle(img, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
        for i in descriptors:
            dist_ = numpy.linalg.norm(i -d_test)
            dist.append(dist_)
    c_d = dict( zip(candidate,dist))

    cd_sorted = sorted(c_d.items(), key = lambda d:d[ 1])

    rec_name = cd_sorted[ 0][ 0]
    cursor.execute("select * from data where name = '%s'" %(rec_name))
    
    cv2.putText(img, rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 255, 255, 255), 2, cv2. LINE_AA)
    img = imutils.resize(img, width = 480)
    img = cv2.cvtColor(img,cv2. COLOR_BGR2RGB)
    cv2.imshow( "Face Recognition", img)
    
    result = cursor.fetchall()
    
    for row in result:
        l2.config(text=str(row))
        l2.place(x=100, y=240)
        

def getuser():
    o=object_text.get() 
    
    cursor.execute("select d.id,name,tel,l.likea from data d,liked l where d.id=l.id AND l.likea like '%s' Order by d.id " %(o))
    result1 = cursor.fetchall()
    col1 = ''
    for col in result1:
        col1+=( str(col) +"\n")
        
    for col in result1:
        
        l2.config(text=str(col1))
        
        l2.place(x=100, y=250)
    
def newdata():
    o1=object_text1.get()
    o2=object_text2.get()
    o3=object_text3.get()
    o4=object_text4.get()
    if (o2=='' and  o3==''):
        list1=[(o1,o4)]
        command = "INSERT INTO liked(id, likea)VALUES(%s, %s)"
        cursor.executemany(command, list1)
        company.commit()
    else:
        list1=[(o1,o2,o3,o4)]
        list2=[(o1,o4)]
        command = "INSERT INTO data(id, name, tel, likea)VALUES(%s, %s, %s, %s)"
        command1 = "INSERT INTO liked(id, likea)VALUES(%s, %s)"
        cursor.executemany(command, list1)
        cursor.executemany(command1, list2)
        company.commit()
def newimage():
    filename=filedialog.askopenfilename()
    img=Image.open('%s'%filename)
    o2=object_text2.get()
    img.save('C:/Users/user/Downloads/Project/rec/%s.jpg'%o2)
def capture():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow("capture", frame)
        key= cv2.waitKey()
        if key== ord("c"):
            cv2.imwrite("output.jpg",frame)
    cap.release()
        cv2.destroyWindow("windows")
       

object_text.place(x=500, y=160)
object_text1.place(x=100, y=60)
l4.place(x=100, y=40)
object_text2.place(x=250, y=60)
l5.place(x=250, y=40)
object_text3.place(x=400, y=60)
l6.place(x=400, y=40)
object_text4.place(x=550, y=60)
l7.place(x=550, y=40)
cursor.execute("SELECT MAX(id) FROM data" )

result = cursor.fetchmany(1)
l8.config(text=str(result[0]))
b=tk.Button(root,width=20,height=2,text='選擇一張圖片', command=openpicture).place(x=70, y=180)


b1=tk.Button(root,width=20,height=2,text="尋找需要此一物品的人",command=getuser).place(x=500, y=180)
b3=tk.Button(root,width=20,height=2,text="新增資料",command=newdata).place(x=550, y=90)
b4=tk.Button(root,width=20,height=2,text="放入圖片",command=newimage).place(x=390, y=90)
b5=tk.Button(root,width=20,height=2,text="開啟webcam拍攝",command=capture).place(x=500, y=300)


tk.mainloop()   
