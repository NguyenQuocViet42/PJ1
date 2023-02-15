from facenet import embeding_face
from PIL import Image
import os
from tkinter import *
from tkinter.ttk import *
from tkinter import ttk
from tkinter import messagebox
import tkinter
import cv2
from PIL import ImageTk, Image
import PCA
import SVM
import ANN
import facenet
import pickle as pk
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import load_model
import numpy as np
from facenet_pytorch import MTCNN
import torch
import sys
import random
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

choose_win = tkinter.Tk()
choose_win.title('Choose the model')
choose_win.geometry('500x250')
choose_win.configure(background='deepskyblue4')
choose = 'FaceNet and SVM'
Text_choose = tkinter.Label(choose_win, text="Choose the Model", foreground="cyan2", font=(
    "Arial Black", 22), background='deepskyblue4')
Text_choose.grid(row=0, column=1)
Text_choose.place(x=115, y=30)

n = tkinter.StringVar()
monthchoosen = ttk.Combobox(
    choose_win, width=27, textvariable=n, font='Arial 15', background='cornsilk1')
next_button = tkinter.Button(choose_win, text="Next", font="Arial 15",
                             command=lambda: choose_win.quit(), width=28, background='cornsilk1')
next_button.grid(column=1, row=9, columnspan=2, sticky=NW)
# Adding combobox drop down list
monthchoosen['values'] = ('FaceNet and SVM', 'FaceNet and ANN Classification',
                          'PCA and SVM', 'PCA and ANN Classification')
monthchoosen.grid(column=1, row=5)
monthchoosen.place(x=100, y=90)
next_button.place(x=100, y=145)
monthchoosen.current(0)
choose_win.mainloop()
choose_ = n.get()

try:
    choose_win.destroy()
except:
    sys.exit(0)
svm = 0
ann = 0

RUN = True
run_detect = 0
num_cam = 0

datafile = open('PJ1\model\PCA_Data.pkl', 'rb')
data = pk.load(datafile)
X = data.X
Y = data.Y
X_train_pca = data.X_train_pca
Y_train = data.Y_train
labelencoder = LabelEncoder()
Y_label = labelencoder.fit_transform(Y)

PCAfile = open('PJ1\model\PCA.pkl', 'rb')
pca = pk.load(PCAfile)

SVMfile = open('PJ1\model\SVM_one.sav', 'rb')
svm_one = pk.load(SVMfile)
SVMfile = open('PJ1\model\SVM_one_fn.sav', 'rb')
svm_one_fn = pk.load(SVMfile)

if choose[0:3] == 'PCA':
    SVMfile = open('PJ1\model\SVM.sav', 'rb')
    svm = pk.load(SVMfile)
    ann = load_model("PJ1\\model\\ann.h5")
else:
    SVMfile = open('PJ1\model\SVM_fn.sav', 'rb')
    svm = pk.load(SVMfile)
    ann = load_model("PJ1\\model\\ann_fn.h5")

import math
def dis_vector(x,y):
    sum = 0
    for i in range(len(x)):
        sum += (x[i]-y[i])**2
    return math.sqrt(sum)

fn_data = open('PJ1\model\FaceNet_Data.pkl','rb')
fn_data = pk.load(fn_data)
X_fn = fn_data.X
X_pca = pca.transform(X)

threshold_svm = open('PJ1\\model\\threshold_svm.txt', 'r')
threshold_svm = threshold_svm.read()
threshold_svm = threshold_svm.split(',')
for i in range(len(threshold_svm)):
    threshold_svm[i] = float(threshold_svm[i])

def find_top2(arr):
    ans = []
    ans.append(np.argmax(arr))
    tmp = arr[ans[0]]
    arr[ans[0]] = 0
    ans.append(np.argmax(arr))
    arr[ans[0]] = tmp
    return ans

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],  
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 
              'kernel': ['rbf', 'sigmoid']} 

def verify_check(x, y, face):
    x_ = np.concatenate( (X_fn[x*20:x*20+20], X_fn[y*20:y*20+20]) , axis=0)
    y_ = [x]*20 +[y]*20
    svm_ = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5, refit = True)
    svm_.fit(x_,y_)
    return np.array(svm_.predict(face))

def SVMpredict(img_test):
    img_test = img_test.reshape(160, 160)
    img_test = img_test.reshape(1, -1)
    img_test = pca.transform(img_test)
    list_predictions = svm.decision_function(img_test.reshape(1,-1))[0]
    top1, top2 = find_top2(list_predictions)
    flag = svm_one.predict(img_test)   
    dis = dis_vector(img_test.reshape(-1,1), X_train_pca[top1*20, :].reshape(-1,1))
    print(dis)
    if dis > 50 or flag == 0:
        return 'Unknown Person', None
    if list_predictions[top1] - list_predictions[top2] > threshold_svm[top1]:
        return labelencoder.inverse_transform(np.array(top1).reshape(1, -1)), None
    else:
        return labelencoder.inverse_transform(verify_check(top1, top2,img_test).reshape(1, -1)), None

def ANNpredict(img_test):
    img_test = img_test.reshape(160, 160)
    img_test = img_test.reshape(1, -1)
    img_test = pca.transform(img_test)
    pr = ann.predict(img_test)
    person_test = np.argmax(pr)
    dis = np.max(pr)
    if dis < 0.6:
        return 'Unknown Person', None
    return labelencoder.inverse_transform(person_test.reshape(1, -1)), dis*100

def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb


count_nhandien = -100
face_dict = {}
arr_dict = []
dis_dict = []
id_dict = []
capture = 0

win = Tk()
win.title("Nhận diện khuôn mặt")
win.geometry("1000x700")

# CONFIG
# PATH
base_dir = os.path.dirname(__file__)
print(base_dir)
# COLOR
lightGray = _from_rgb((240, 240, 240))
white = _from_rgb((255, 255, 255))

# FONT
font_header1 = "Arial 20 bold"
font_header2 = "Arial 16 bold"
font_content = "Arial 12"
#
def SVMpredict_fn(face):
    face = Image.open('tmp.jpg')
    face = embeding_face(face)
    img_test = face.reshape(1, -1)
    person_test = svm.predict(img_test)
    dis = dis_vector(img_test.reshape(-1,1), X_fn[person_test*20, :].reshape(-1,1))
    flag = svm_one_fn.predict(img_test)  
    if dis > 0.9 or flag == 0:
        return 'Unknown Person', dis
    return labelencoder.inverse_transform(person_test), dis


def ANNpredict_fn(face):
    face = Image.open('tmp.jpg')
    face = embeding_face(face)
    img_test = face.reshape(1, -1)
    pr = ann.predict(img_test)
    person_test = np.argmax(pr)
    dis = np.max(pr)
    if dis < 0.6:
        return 'Unknown Person', None
    return labelencoder.inverse_transform(person_test.reshape(1, -1)), dis*100

model_pre = 0
if choose == 'FaceNet and SVM':
    model_pre = SVMpredict_fn
elif choose == 'FaceNet and ANN Classification':
    model_pre = ANNpredict_fn
elif choose == 'PCA and SVM':
    model_pre = SVMpredict
else:
    model_pre = ANNpredict

# IMAGE
bg_image = Image.open(base_dir+"//imageGUI//bg_app.jpg")
bg_image = bg_image.resize(
    (1000, 700), Image.ANTIALIAS)
bg_image = ImageTk.PhotoImage(bg_image)

default_them_nguoi = Image.open(base_dir+"//imageGUI//default_Image.png")
default_them_nguoi = default_them_nguoi.resize(
    (560, int(3*560/4)), Image.ANTIALIAS)
default_them_nguoi = ImageTk.PhotoImage(default_them_nguoi)

default_empty = Image.open(base_dir+"//imageGUI//default_empty.png")
default_empty = default_empty.resize(
    (60, 60), Image.ANTIALIAS)
default_empty = ImageTk.PhotoImage(default_empty)

button_them_nguoi = Image.open(base_dir+"//imageGUI//button_them_nguoi.png")
button_them_nguoi = button_them_nguoi.resize(
    (60, 60), Image.ANTIALIAS)
button_them_nguoi = ImageTk.PhotoImage(button_them_nguoi)

arow = Image.open(base_dir+"//imageGUI//arow.png")
arow = arow.resize(
    (160, 80), Image.ANTIALIAS)
arow = ImageTk.PhotoImage(arow)
# End config

trang_chu = tkinter.Frame(win)
nhan_dien = tkinter.Frame(win)
them_nguoi = tkinter.Frame(win)

frames = (trang_chu, nhan_dien, them_nguoi)
for f in frames:
    f.place(relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)


def switch(frame):
    for f in frames:
        for widget in f.winfo_children():
            widget.destroy()
    if (frame == trang_chu):
        trangChu()
    elif (frame == nhan_dien):
        global run_detect
        run_detect = 1
        nhanDien()
    elif (frame == them_nguoi):
        reRenderImageButton()
        themNguoi()
    frame.tkraise()


def trangChu():
    f_trang_chu = tkinter.Frame(
        trang_chu, padx=0, pady=0, bg='lightblue')
    f_trang_chu.place(
        relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)
    f_trang_chu.grid_columnconfigure(0, weight=1)
    f_trang_chu.grid_columnconfigure(1, weight=1)
    f_trang_chu.grid_columnconfigure(2, weight=1)
    f_trang_chu.grid_columnconfigure(3, weight=1)
    f_trang_chu.grid_columnconfigure(4, weight=1)
    f_trang_chu.grid_columnconfigure(5, weight=1)
    f_trang_chu.grid_rowconfigure(0, weight=1)
    f_trang_chu.grid_rowconfigure(1, weight=1)

    tkinter.Label(f_trang_chu, image=bg_image, anchor=W).place(
        relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)

    tkinter.Label(f_trang_chu, text="Chọn chức năng", font=font_header1, bg='#1CA7E4', fg="white").grid(
        column=0, row=0, columnspan=6)
    tkinter.Button(f_trang_chu, text="Nhận diện", font=font_header2,  bg="#1AAAEA", fg="white", command=lambda: switch(
        nhan_dien)).grid(column=2, row=1, columnspan=1, sticky=N)
    tkinter.Button(f_trang_chu, text="Thêm người", font=font_header2, bg="#1AAAEA", fg="white", command=lambda: switch(
        them_nguoi)).grid(column=3, row=1, columnspan=1, sticky=N)

person_pre = 'Unknown Person'
img_person = Image.open(base_dir + '\\Face_image\\Person\\'+person_pre+'.jpg')
img_person = img_person.resize((200, 200), Image.ANTIALIAS)
img_ = ImageTk.PhotoImage(img_person)

def nhanDien():
    f_nhan_dien = tkinter.Frame(nhan_dien)
    f_nhan_dien.place(
        relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)
    f_nhan_dien.grid_columnconfigure(0, weight=1)
    # f_nhan_dien.grid_columnconfigure(1, weight=1)
    f_nhan_dien_left = tkinter.Frame(
        f_nhan_dien, bg=lightGray, padx=20, pady=5)
    f_nhan_dien_left.place(
        relx=0, rely=0, relheight=1, relwidth=0.6, anchor=NW)

    f_nhan_dien_right = tkinter.Frame(f_nhan_dien, bg=white, padx=30, pady=5)
    f_nhan_dien_right.place(
        relx=1, rely=0, relheight=1, relwidth=0.4, anchor=NE)

    f_nhan_dien_left.grid_columnconfigure(0, weight=1)
    f_nhan_dien_left.grid_columnconfigure(1, weight=1)
    f_nhan_dien_left.grid_rowconfigure(0, weight=1)
    f_nhan_dien_left.grid_rowconfigure(1, weight=1)
    f_nhan_dien_left.grid_rowconfigure(2, weight=1)
    f_nhan_dien_left.grid_rowconfigure(3, weight=9)
    f_nhan_dien_left.grid_rowconfigure(4, weight=3)
    # f_nhan_dien_right.grid_columnconfigure(0, weight=1)
    # f_nhan_dien_right.grid_columnconfigure(1, weight=4)
    # f_nhan_dien_right.grid_columnconfigure(2, weight=1)
    # f_nhan_dien_right.grid_rowconfigure(0, weight=1)
    # f_nhan_dien_right.grid_rowconfigure(1, weight=1)
    # f_nhan_dien_right.grid_rowconfigure(2, weight=1)
    # f_nhan_dien_left.grid_rowconfigure(4, weight=5)

    tkinter.Button(f_nhan_dien_left, text="Trở về", font=font_content, command=lambda: switch(
        trang_chu)).grid(column=0, row=0, columnspan=2, sticky=NW)
    tkinter.Label(f_nhan_dien_left, text="Nhận diện khuôn mặt",
                  font=font_header1, anchor=W).grid(column=0, row=1, columnspan=2, sticky=W)
    tkinter.Label(f_nhan_dien_left,
                  text="Đưa mặt vào trước camera để nhận diện",
                  font=font_content, anchor=W, wraplength=500, justify=LEFT).grid(row=2, column=0, columnspan=2, sticky=NW)
    camera = tkinter.Label(f_nhan_dien_left, text="", image=default_them_nguoi)
    camera.grid(column=0, row=3, columnspan=2, sticky=NW)
    tkinter.Button(f_nhan_dien_left, text="Bắt đầu nhận diện", font=font_header2, bg="blue", fg="white",
                   
                   command=lambda: start_nhandien()).grid(column=0, row=4, columnspan=2, sticky=N, padx=5, pady=5)
    img_label = tkinter.Label(f_nhan_dien_right, image=img_)
    img_label.place(relx=0.5, rely=0.3,  anchor=N)
    name_label = tkinter.Label(f_nhan_dien_right, text=person_pre, font=font_header1, anchor=W)
    name_label.place(relx=0.5, rely=0.6, anchor=N)
    
    global run_detect, count_nhandien, face_dict
    run_detect = 1
    count_nhandien = -100
    face_dict = {}
    arr_dict = []
    dis_dict = []
    id_dict = []
    camera_nhandien(camera, name_label, img_label, predicter=model_pre)

    def start_nhandien():
        global count_nhandien, face_dict, arr_dict, dis_dict, id_dict
        count_nhandien = 11
        face_dict = {}
        arr_dict = []
        dis_dict = []
        id_dict = []
    # RIGHT

def takeAPhoto():
    global capture
    capture = 20

def themNguoi():
    f_them_nguoi = tkinter.Frame(them_nguoi)
    f_them_nguoi.place(
        relx=0, rely=0, relheight=1, relwidth=1, anchor=NW)

    f_them_nguoi_left = tkinter.Frame(
        f_them_nguoi, bg=lightGray, padx=20, pady=5)
    f_them_nguoi_left.place(
        relx=0, rely=0, relheight=1, relwidth=0.6, anchor=NW)

    f_them_nguoi_right = tkinter.Frame(f_them_nguoi, bg=white, padx=30, pady=5)
    f_them_nguoi_right.place(
        relx=1, rely=0, relheight=1, relwidth=0.4, anchor=NE)

    f_them_nguoi_left.grid_columnconfigure(0, weight=1)
    f_them_nguoi_left.grid_columnconfigure(1, weight=1)
    f_them_nguoi_left.grid_rowconfigure(0, weight=1)
    f_them_nguoi_left.grid_rowconfigure(1, weight=1)
    f_them_nguoi_left.grid_rowconfigure(2, weight=1)
    f_them_nguoi_left.grid_rowconfigure(3, weight=9)
    f_them_nguoi_left.grid_rowconfigure(4, weight=5)

    tkinter.Button(f_them_nguoi_left, text="Trở về", font=font_content, command=lambda: switch(
        trang_chu)).grid(column=0, row=0, columnspan=2, sticky=NW)
    tkinter.Label(f_them_nguoi_left, text="Nhận diện khuôn mặt",
                  font=font_header1, anchor=W).grid(column=0, row=1, columnspan=2, sticky=W)
    tkinter.Label(f_them_nguoi_left,
                  text="Để thêm một khuôn mặt mới, nhấn vào biểu tượng dấu cộng ở màn hình bên tay phải",
                  font=font_content, anchor=W, wraplength=500, justify=LEFT).grid(row=2, column=0, columnspan=2, sticky=NW)
    camera = tkinter.Label(f_them_nguoi_left, text="",
                           image=default_them_nguoi)
    camera.grid(
        column=0, row=3, columnspan=2, sticky=NW)

    captureButton = tkinter.Button(
        f_them_nguoi_left, text="Chụp ảnh", font=font_header2, command=takeAPhoto)
    captureButton.grid(column=0, row=4, columnspan=1, sticky=N)

    finishButton = tkinter.Button(
        f_them_nguoi_left, text="Re-Training", font=font_header2, command=lambda: endVideo(camera))
    finishButton.grid(column=1, row=4, columnspan=1, sticky=N)

    # RIGHT
    f_them_nguoi_right.grid_columnconfigure(0, weight=1)
    f_them_nguoi_right.grid_columnconfigure(1, weight=1)
    f_them_nguoi_right.grid_columnconfigure(2, weight=1)
    f_them_nguoi_right.grid_rowconfigure(0, weight=2)
    f_them_nguoi_right.grid_rowconfigure(1, weight=2)
    f_them_nguoi_right.grid_rowconfigure(2, weight=2)
    f_them_nguoi_right.grid_rowconfigure(3, weight=3)

    tkinter.Label(f_them_nguoi_right, text="Thêm khuôn mặt mới", bg=white,
                  font=font_header2, fg='lightblue', justify=CENTER).grid(column=0, row=0, columnspan=3, sticky=S)
    tkinter.Label(f_them_nguoi_right, image=arow, bg=white, justify=CENTER).grid(
        column=0, row=1, columnspan=3, sticky=W)

    tkinter.Button(f_them_nguoi_right, image=button_them_nguoi, relief=FLAT, command=lambda: getName(camera)).grid(
        column=0, row=2, columnspan=1, sticky=NW)
    global capture
    capture = - 20
    listButton = []
    for i in range(5):
        listButton.append(tkinter.Button(f_them_nguoi_right,
                          image=listImage[i], relief=FLAT))
    '''Set image for button'''

    listButton[0].grid(column=1, row=2, columnspan=1, sticky=N)
    listButton[1].grid(column=2, row=2, columnspan=1, sticky=NE)
    listButton[2].grid(column=0, row=2, columnspan=1, sticky=SW)
    listButton[3].grid(column=1, row=2, columnspan=1, sticky=S)
    listButton[4].grid(column=2, row=2, columnspan=1, sticky=SE)


listImage = [default_empty, default_empty,
             default_empty, default_empty, default_empty]


def reRenderImageButton():
    base_path_image = base_dir+"//Face_image//Caltech_face//"
    folder_image = [os.path.join(base_path_image, f)
                    for f in os.listdir(base_path_image)]
    len_i = len(folder_image)
    i = 1
    while i < 6:
        path_folder = folder_image[len_i-i]
        if not len(os.listdir(path_folder)):
            continue
        else:
            image_path = os.path.join(
                path_folder, os.listdir(path_folder)[0])
            load_img = (Image.open(
                image_path))
            load_img = load_img.resize(
                (60, 60), Image.ANTIALIAS)
            listImage[i-1] = ImageTk.PhotoImage(load_img)
        i += 1


'''CAMERA'''
cap = cv2.VideoCapture(0)
start = False
name = ""
container_folder = ""
count = 0

# Define function to show frame

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True,
              post_process=False, device=device)


def show_frames(camera):
    global capture, container_folder, count, name
    if start == False:
        return
    # Get the latest frame and convert into Image
    frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
    # cap window
    if capture > 0:
        try:
            os.chdir('PJ1\\Face_image\\Caltech_face\\' + name)
            print(name)
        except:
            pass
        # tạo tên ảnh
        boxes, _, points_list = mtcnn.detect(frame, landmarks=True)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int, box.tolist()))
                try:
                    face = mtcnn(frame, str(capture)+".jpg")
                    capture -= 1
                    cv2.putText(frame, str(round((20-capture-1) / 20 * 100, 2))+'%',
                                (bbox[0]+70, bbox[1]-20), cv2.FONT_HERSHEY_DUPLEX, 1, (239, 50, 239), 2, cv2.LINE_AA)
                except Exception as e:
                    print(e)
                frame = cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
        # plt.imshow(photoSave, cmap='gray')
        # get img capture
        # end capture
    # Repeat after an interval to capture continiously
    img = Image.fromarray(frame)
    img = img.resize((560, int(3*560/4)), Image.ANTIALIAS)

    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(img)
    camera.imgtk = imgtk
    camera.configure(image=imgtk)
    if capture <= 0 and capture >= -10:
        capture = -100
        messagebox.showinfo("Thông báo", "Thêm người thành công!!")
        os.chdir(base_dir.replace('\PJ1',''))
        switch(trang_chu)
        return
    camera.after(5, lambda: show_frames(camera))

def camera_nhandien(camera, name_label, img_label,predicter):
    face = []
    global count_nhandien, face_dict, arr_dict, dis_dict, id_dict
    # Get the latest frame and convert into Image
    frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB) # Lấy frame từ camera
    if count_nhandien > 0:  # Kiểm tra xem chụp đủ 10 ảnh chưa
        boxes, _, points_list = mtcnn.detect(frame, landmarks=True) # Detect vị trí mặt
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int, box.tolist()))
                try:
                    face_ = mtcnn(frame, "tmp.jpg")     # Lưu lại khuôn mặt
                    face = cv2.imread("tmp.jpg", 0)     # Chuyển về kênh màu Gray
                    face_ = cv2.imread("tmp.jpg")
                    id, dis = predicter(face)           # Truyền khuôn mặt vào hàm nhận diện
                    try:
                        os.remove('tmp.jpg')
                    except:
                        pass
                    face = cv2.resize(src=face, dsize=(160, 160))
                    cv2.putText(frame, str(round((11-count_nhandien+1) / 11 * 100, 2))+'%',
                                (bbox[0]+20, bbox[1]-20), cv2.FONT_HERSHEY_DUPLEX, 1, (239, 50, 239), 2, cv2.LINE_AA)   # In ra số ảnh đã chụp
                    id = str(id)
                    # Đếm số lần được dự đoán của các nhãn
                    if id != 'Unknown Person':
                        arr_dict.append(face_)
                        dis_dict.append(dis)
                        id_dict.append(id)
                    if id in face_dict.keys():
                        face_dict[id] += 1
                        count_nhandien -= 1
                    else:
                        face_dict[id] = 1
                        count_nhandien -= 1
                except Exception as e:
                    print(e)
                frame = cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)

    img = Image.fromarray(frame)
    img = img.resize((560, int(3*560/4)), Image.ANTIALIAS)
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(img)
    camera.imgtk = imgtk
    camera.configure(image=imgtk)
    if count_nhandien <= 0 and count_nhandien >= -10:
        count_nhandien = -100
        max = 0
        name = ''
        for i in face_dict.keys():
            if face_dict[i] > max:
                max = face_dict[i]
                name = i
        id_name = name
        name = name.replace('[' , '')
        name = name.replace(']' , '')
        name = name.replace('\'' , '')
        img_person = Image.open(base_dir + '\\Face_image\\Person\\'+name+'.jpg')
        img_person = img_person.resize((200, 200), Image.ANTIALIAS)
        img_person = ImageTk.PhotoImage(img_person)
        name_label['text'] = name
        img_label['image'] = img_person
        question = messagebox.askquestion("KẾT QUẢ NHẬN DIỆN", 'Đây có phải là?\n' + name)
        if name != 'Unknown Person' and question != 'no':
            min = 999
            face_min = 0
            for i in range(len(id_dict)):
                if id_dict[i] == id_name:
                    if min > dis_dict[i]:
                        min = dis_dict[i]
                        face_min = arr_dict[i]
            container_folder = base_dir + "//Face_image//Caltech_face//" + name
            list_img = os.listdir(container_folder)
            index = random.randint(1,21)
            ran_name = list_img[index]
            os.chdir(container_folder)
            cv2.imwrite(ran_name,face_min)
            os.chdir(base_dir.replace('\PJ1',''))
    camera.after(5, lambda: camera_nhandien(camera, name_label, img_label,predicter=model_pre))


def startVideo(camera):
    global start
    start = True
    show_frames(camera)


def endVideo(camera):
    global start, pca, svm, ann, labelencoder, svm_one, svm_one_fn, X_fn
    camera['image'] = default_them_nguoi
    start = False
    messagebox.showinfo(
        "Waiting", "Re_training Starting, Please wait a minute")
    print(os. getcwd())
    PCA.PCA()
    datafile = open('PJ1\model\PCA_Data.pkl', 'rb')
    data = pk.load(datafile)
    PCAfile = open('PJ1\model\PCA.pkl', 'rb')
    pca = pk.load(PCAfile)
    Y = data.Y
    X_train_pca = data.X_train_pca
    Y_train = data.Y_train
    labelencoder = LabelEncoder()
    onehotencoder = OneHotEncoder(sparse=False)
    labelencoder.fit(Y)
    arr = np.arange(0, X_train_pca.shape[0]//19)
    onehotencoder.fit(arr.reshape(-1, 1))
    Y_train_label = onehotencoder.transform(Y_train.reshape(-1, 1))

    facenet.Face_Net(Y_train, Y_train_label)

    SVM.SVM(X_train_pca, Y_train, 'SVM.sav')

    ANN.ANN(X_train_pca, Y_train_label, 'ann.h5')
    
    X_more = data.X_more
    X_more = pca.transform(X_more)
    X_ = np.concatenate((X_train_pca, X_more), axis=0)
    Y_ = [1]*X_train_pca.shape[0] + [0]*X_more.shape[0]
    SVM.SVM(X_, Y_, 'SVM_one.sav')
    
    SVMfile = open('PJ1\model\SVM_one.sav', 'rb')
    svm_one = pk.load(SVMfile)
    SVMfile = open('PJ1\model\SVM_one_fn.sav', 'rb')
    svm_one_fn = pk.load(SVMfile)
    
    fn_data = open('PJ1\model\FaceNet_Data.pkl','rb')
    fn_data = pk.load(fn_data)
    X_fn = fn_data.X
    
    if choose[0:3] == 'PCA':
        print('PCA')
        SVMfile = open('PJ1\model\SVM.sav', 'rb')
        svm = pk.load(SVMfile)
        ann = load_model("PJ1\\model\\ann.h5")
    else:
        print('FaceNet')
        SVMfile = open('PJ1\model\SVM_fn.sav', 'rb')
        svm = pk.load(SVMfile)
        ann = load_model("PJ1\\model\\ann_fn.h5")

    messagebox.showinfo("Successful :>", "Re_training Completed <3")
    switch(trang_chu)

def save(file_name, img, path):
    # Set vị trí lưu ảnh
    os.chdir(path)
    # Lưu ảnh
    cv2.imwrite(file_name, img)


def getName(camera):
    top = tkinter.Toplevel(win)

    top.title("window")
    top.geometry("230x100")

    label = tkinter.Label(top, text="Nhập tên:", font=font_header1)
    label.place(relx=0.5, rely=0.2, anchor=N)

    text = tkinter.Text(top, height=1, width=20)
    text.place(relx=0.5, rely=0.5, anchor=N)

    def get():
        global container_folder, count, name
        name = text.get(1.0, END)[0:-1]
        container_folder = base_dir + "//Face_image//Caltech_face//" + name
        if not os.path.exists(container_folder):
            os.makedirs(container_folder)
            os.chdir(container_folder)
            count = 0
        else:
            count = len(os.listdir(container_folder))
        startVideo(camera)
        top.destroy()

    button = tkinter.Button(top, text="OK", command=get)
    button.place(relx=0.5, rely=0.8, anchor=N)


switch(trang_chu)
win.mainloop()