import os
import glob
import shutil
from shutil import copyfile
import pandas as pd
from imread import imread, imsave
from matplotlib import pyplot as plt
directory1 = "/media/eeglab/YG_Storage/CT_Xray/*/*/"
file1 = "./data/xray/Cardio_Edema_Train.txt"
directory2 = "/media/eeglab/YG_Storage/CT_Xray/images/"
index = pd.read_csv(file1, delim_whitespace=True, header=1)
# a = index.index[0]
# b = glob.glob(os.path.join(directory1, a))

def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1

for i in range(0, 2001):
    a = index.index[i]
    b = glob.glob(os.path.join(directory1, a))
    c = listToString(b)
    shutil.copy(c, directory2)
