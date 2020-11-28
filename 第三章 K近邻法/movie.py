#使用k近邻算法进行电影的分类
import pandas as pd
rowdata={'电影名字':['无问西东','后来的我们','前任3','红海行动','唐人街探案','战狼2'],
         '打斗镜头':[1,5,12,108,112,115],
          '接吻镜头':[101,89,97,5,9,8],
          '电影类型':['爱情片','爱情片','爱情片','动作片','动作片','动作片']}
movie_data=pd.DataFrame(rowdata)
print(movie_data)
#计算已知类别数据集中的点到当前点的距离
new_data=[24,67]
dist=list((((movie_data.iloc[:6,1:3]-new_data)**2).sum(1))**0.5)#0代表行相交 1代表列相交
movie_data.iloc[:6,1:3] #前六行 1到3列
#print(movie_data.iloc[:6,1:3])
print(dist)
#对距离进行升序排序 选取距离最小的4个点
k=4
dist_l=pd.DataFrame({'dist': dist,'label':movie_data.iloc[:6,3]})
dr=dist_l.sort_values(by='dist')[:k]#sort_values pands 里面的排序函数  [:k]进行接片
print(dr)
#进行频率
re=dr.loc[:,'label'].value_counts()
re.index
print(re.index)
print(re.index[0])
#
result=[]
result.append(re.index[0])
result

#封装函数
