import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
import numpy as np
import glob
import pickle as pk
import SVM
import ANN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            fixed_image_standardization
        ])
    return transform(img)

def fixed_image_standardization(image_tensor):
    processed_tensor = image_tensor
    return processed_tensor

model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)

model.eval()

def embeding_face(img):
    return (model(trans(img).to(device).unsqueeze(0))).detach().numpy()[0,:]

def split_data(X):
    split_index = list(range(0,X.shape[0], 20))
    X_test = X[split_index]

    # preprare training set by removing items in test set
    X_train = X.copy()
    for i in split_index[::-1]:
        X_train  = np.delete(X_train,i,axis = 0 )
    print('X_train: ', X_train.shape)
    print('X_test: ', X_test.shape)
    return X_train, X_test

class FaceNet_Data():
    def __init__(self, X, X_train, X_test):
        self.X = X
        self.X_train = X_train
        self.X_test = X_test

class Raw_Data():
    def __init__(self, X):
        self.X = X

def Face_Net(Y_train, Y_train_label):
    print('Running FaceNet')    
    X_facenet = []
    list_link = glob.glob('PJ1\\Face_image\\Caltech_face\\' + "*")
    datafile = open('PJ1\model\FaceNet_RawData.pkl', 'rb')
    data = pk.load(datafile)
    X_more = data.X
    for link in list_link:
        for filename in glob.glob( link + '\\' + '*.jpg'): 
            img = Image.open(filename)
            x = (model(trans(img).to(device).unsqueeze(0))).detach().numpy()[0,:]
            X_facenet.append(x)
    X_facenet = np.array(X_facenet)
    X_ = np.concatenate((X_facenet, X_more), axis=0)
    Y_ = [1]*X_facenet.shape[0] + [0]*X_more.shape[0]
    X_train, X_test = split_data(X_facenet)
    data = FaceNet_Data(X_facenet, X_train, X_test)
    pk.dump(data, open("PJ1\\model\\FaceNet_Data.pkl","wb"))
    SVM.SVM(X_train,Y_train, 'SVM_fn.sav')
    SVM.SVM(X_, Y_, 'SVM_one_fn.sav')
    ANN.ANN_fn(X_train, Y_train_label,'ann_fn.h5')
    print('Done FaceNet')
    print('___________________________________________')