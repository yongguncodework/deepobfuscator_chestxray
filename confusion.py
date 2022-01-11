print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import resnet
import seaborn as sns
from scipy.io import loadmat
# from sklearn import svm, datasets
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names
# cl = np.array(['Cardiomegaly','Edema'])

data_file_name = 'Chest_Xray_15disease_Label.mat'
data_file_name2 = 'Chest_Xray_15disease_Label_sci.mat'
DATA_FOLDER_PATH = '/media/eeglab/YG_Storage/CT_Xray'
label_path = '/media/eeglab/YG_Storage/'
FILE_PATH = label_path + '/' + data_file_name
FILE_PATH2 = label_path + '/' + data_file_name2
img1 = np.array(np.load("/home/eeglab/PycharmProjects/YG/Lung_data/CT_Xray/Chest_Xray_15disease_images.npy"))
img2 = img1[:,::4,::4]

mat = loadmat(FILE_PATH2)

age = np.array(mat['age'])
disease = np.array(mat['id'])
gender = np.array(mat['gender'])
image = np.array(mat['image'])
image_sci = (mat['image'])
pred = np.load('CT_Xray/keras_15diseaseypreds.npy')
X_train, X_test, y_train, y_test = train_test_split(img2, disease, test_size=0.20, random_state=42)

# Y_test = np_utils.to_categorical(y_test, 16)
Y_test = y_test
Y_pred = np.argmax(pred, axis=1)

print('Confusion_matrix')
print(confusion_matrix(Y_test, Y_pred))
print('classification report')
target_names = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
                               'Emphysema','Fibrosis','Hernia','Infiltration','Mass','No Finding',
                               'Nodule', 'Pleural Thickening', 'Pneumonia','Pneumothorax']
print(classification_report(Y_test, Y_pred, target_names=target_names))

cm = confusion_matrix(Y_test, Y_pred)

cmdf = pd.DataFrame(cm, index=['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
                               'Emphysema','Fibrosis','Hernia','Infiltration','Mass','No Finding',
                               'Nodule', 'Pleural Thickening', 'Pneumonia','Pneumothorax'],
                    columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
                               'Emphysema','Fibrosis','Hernia','Infiltration','Mass','No Finding',
                               'Nodule', 'Pleural Thickening', 'Pneumonia','Pneumothorax'])

plt.figure()
sns.heatmap(cmdf, annot=True)
plt.title('ResNet-50 on Disease prediction \nAUC:{0:.3f}'.format(0.793))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

