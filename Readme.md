# 决策树实现对鸢尾花数据集的分类

- ## 代码实现

```python
from sklearn import tree,datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
import graphviz

iris = datasets.load_iris()		# 导入鸢尾花数据集
x = iris.data					# 划分数据和标签
y = iris.target
clf = DecisionTreeClassifier(random_state=25,	# 实例化决策树模型
                             max_depth=4,
                             min_samples_split=3,
                             min_samples_leaf=2
                             )
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3)	# 划分训练集和测试集
clf = clf.fit(Xtrain, Ytrain)										# 训练集拟合模型
score_ = clf.score(Xtest, Ytest)									# 计算精确度
print(score_)
score = cross_val_score(clf,x,y,cv=10).mean()						# 交叉验证
print(score)
feature_name = ['萼片长度','萼片宽度','花瓣长度','花瓣宽度']
dot_data = tree.export_graphviz(clf,
                                feature_names = feature_name,
                                class_names=["Iris Setosa","Iris Versicolour","Iris Virginica"],
                                filled=True,
                                rounded=True,
                                fontname="Microsoft YaHei"
                               )
graph = graphviz.Source(dot_data)
graph.view()		# 绘制决策树
```
一开始我在绘制决策树时出现了中文乱码，查了一下后发现将fontname参数改为Microsoft YaHei即可解决。

![image](https://github.com/bnsanshi/Classification-of-iris-datasets/blob/main/Source.gv_00.jpg)

- ## 思路
  实例化模型➖调参➖训练模型➖测试模型  
  
  调参过程：绘制学习曲线，根据曲线调整合适的参数

  ![image](https://github.com/bnsanshi/Classification-of-iris-datasets/blob/main/max_depth.png)


![image](https://github.com/bnsanshi/Classification-of-iris-datasets/blob/main/min_samples_split.png)

![image](https://github.com/bnsanshi/Classification-of-iris-datasets/blob/main/min_samples_leaf.png)



决策树在鸢尾花数据集上的表现似乎还不错，其实好像不需要怎么调参。




