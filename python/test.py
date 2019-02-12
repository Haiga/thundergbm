from thundergbm_scikit import *
from sklearn.datasets import *

x,y = load_svmlight_file("../dataset/test_dataset.txt")
clf = TGBMModel()
clf.fit(x,y)

x2,y2=load_svmlight_file("../dataset/test_dataset.txt")
y_predict=clf.predict(x2, y2)
