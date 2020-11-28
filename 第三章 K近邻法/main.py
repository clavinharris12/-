#使用sklearn中KNN算法识别 Iris（鸢尾花）类别
import numpy as np
from sklearn import datasets  # 从sklearn自带数据库中加载鸢尾花数据
from sklearn.model_selection import train_test_split  # 引入train_test_split函数
from sklearn.neighbors import KNeighborsClassifier  # 引入KNN分类器
from sklearn.metrics import accuracy_score#计算准确度
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns #画图
iris = datasets.load_iris()  # 将鸢尾花数据存在iris中
iris_X = iris.data  # 指定训练数据iris_X
iris_y = iris.target  # 指定训练目标iris_y
#print(iris_X[:2,:])   #查看前两个例子的所有特征值
#print(iris_y)#/查看目标标签名称
# 使用train_test_split（）函数将数据集分成用于训练的data和用于测试的data
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
knn = KNeighborsClassifier()  # 调用KNN分类器
knn.fit(X_train, y_train)  # 训练KNN分类器
y_pre=(knn.predict(X_test))#预测值
class_names = iris.target_names
#混沌矩阵
def show_confusion_matrix(validations, predictions):#定义函数
  from sklearn.metrics import confusion_matrix #引入模块
  matrix = confusion_matrix(validations, predictions)
  plt.figure(figsize=(6,4))
  sns.heatmap(matrix, cmap='Greens', linecolor='black', linewidth='1', xticklabels=class_names, yticklabels=class_names, annot=True, fmt='d')
  plt.title("Confusion Matrix")
  plt.ylabel('True Label')
  plt.xlabel('Predicted Label')
  plt.show()
show_confusion_matrix(y_pre, y_test)
print('Accuracy: %.2f' % accuracy_score(y_pre,y_test))

