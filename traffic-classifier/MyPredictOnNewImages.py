import glob
import os
import cv2
import sys

from MyTrainingModelWrapper import MyTrainingModelWrapper


images=[]
labels=[]

pred_labels={"pred1_img.png":11,"pred2_img.png":1,"pred3_img.png":12,"pred5_img.png":38,"pred6_img.png":34,"pred8_img.png":18,"pred9_img.png":25,"pred10_img.png":3 }

for i, fpath in enumerate(glob.glob('new_pred_images/*_img.png')):
    file_name=os.path.basename(fpath)
    images.append(cv2.imread(fpath))
    labels.append(pred_labels[file_name])

print(labels)

mytraining_model_wrapper = MyTrainingModelWrapper()
mytraining_model_wrapper.predict(images, labels)
sys.exit(0)