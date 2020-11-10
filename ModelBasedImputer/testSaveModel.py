from sklearn import datasets
import pickle
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#分割数据集
data_x,data_y = datasets.load_iris(return_X_y=True)
train_X,test_X,train_y,test_y = train_test_split(data_x,data_y,test_size=0.2,random_state=2,stratify=data_y)
#训练模型
lr = LinearRegression()
lr.fit(train_X,train_y)

#将训练的模型保存
direction = joblib.dump(lr,"ModelParams/test.joblib")
testy = lr.predict(test_X)
print(testy)

#下载模型
lr2 = joblib.load("ModelParams/test.joblib")
#模型预测
testy2 = lr2.predict(test_X)
print(testy2)


#重新设置模型参数并训练
lr.set_params(normalize=True).fit(train_X,train_y)
#新模型做预测
lr.predict(test_X)