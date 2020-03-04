import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
x_t, x_val, y_t, y_val = train_test_split(data, target, train_size=0.8)
x_train, x_test, y_train, y_test = train_test_split(x_t, y_t, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_val= scaler.transform(x_val)
cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]
Data_train=x_train@vectores.T
Data_test=x_test@vectores.T
Data_val=x_val@vectores.T
Data_train=Data_train[:,:10]
Data_test=Data_test[:,:10]
Data_val=Data_val[:,:10]
Score=np.array([])
Cvs=np.linspace(-2,2,300)
VC=10**Cvs
for Cv in VC:
    clf=SVC(C=Cv,kernel='linear')
    clf.fit(Data_train,y_train)
    yt_pred= clf.predict(Data_test)
    F1=np.zeros(10)
    for i in range(10):
        y_testf=y_test==i
        yt_predf=yt_pred==i
        F1[i]=f1_score(y_testf,yt_predf)
    Score=np.append(Score,np.sum(F1)/10) 
MatConf=np.zeros((10,10))
VCO=VC[np.argmax(Score)]
clf=SVC(C=VCO,kernel='linear')
clf.fit(Data_train,y_train)
yval_pred=clf.predict(Data_val)
for i in range(10):
    for i2 in range(10):
        MatConf[i,i2]=np.sum((y_val==i)*(yval_pred==i2))
plt.figure(figsize=(10,10))
CNames=["0","1","2","3","4","5","6","7","8","9"]
Ejes=["Truth","Predict"]
plt.imshow(MatConf)
plt.ylabel("Truth")
plt.xlabel("Predict")
plt.xticks(np.arange(10),CNames)
plt.yticks(np.arange(10),CNames)
for i in range(10):
    for i2 in range(10):
        plt.text (i-0.3,i2,"%2.3f"%(MatConf[i,i2]/np.sum(MatConf[:,i2])))
plt.title("C=%4.3f"%VCO)
plt.savefig("Matriz_SVM.png")