#TODO 部分参考https://github.com/sjtupig/MissingImputer
# 本文件定义多目标数据缺失填充xgboost类，针对每一个属性，使用其他属性作为输入，进行提升树的训练。

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
# from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer as Imputer #简单的插值方法，
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import xgboost as xgb 
import lightgbm as lgb
import joblib
from hyperopt import hp
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import pandas as pd
import random
random.seed(1)
import yaml
import json #用于参数文件的存储与读取
class MultiXGboost():
	def __init__(self,params, max_iter = 6, ini_fill = True, ini_strategy_reg = 'mean',
		tol = 1e-3, model_reg = "xgboost",model_clf = "xgboost"):
		'''
		-max_iter:迭代次数
		-ini_fill：是否要进行简单填补(False仅对xgb和lgb有效)
		-ini_strategy_reg:连续变量简单填补规则, mean or median
		-ini_strategy_clf:离散变量简单填补规则, only most_frequent
		-cat_index:离散变量索引(int)
		-tol:阈值
		-model_reg:连续变量采用的预测缺失值模型, be xgboost,lightgbm, randomforest, knn
		-model_clf:离散变量采用的预测缺失值模型
		'''
		self.params = params # 模型参数
		self.best_params = params #用于存储最佳的模型参数
		self.ini_fill = ini_fill  #是否简单初始化缺失部分。
		self.max_iter = max_iter  #最大迭代次数，每次迭代，缺失部分都会被修改。
		self.imputer_reg = Imputer(strategy = ini_strategy_reg)  #TODO 使用均值预填充
		self.tol = tol  #误差阈值，小于阈值则停止训练
		self.model_reg = model_reg #回归模型
		#TODO 记录默认均值

	def fit(self, X,cat_list=[]):

		#TODO 训练模型，记录默认填充的均值
		# X为训练集矩阵。遍历每一列为特定估计器的目标输出，其他列为输入
		# cat_list 类别属性所在列
		# -model_params:params for models,it should be a map

		X = check_array(X, dtype=np.float64, force_all_finite=False) #核对矩阵，判断数据类型
		self.params_mean = np.nanmean(X,axis=0) #逐行求变量的均值  #TODO 实时系统中完成
		self.cat_list = cat_list
		if X.shape[1] == 1: # 需要多维的数据
			raise ValueError("your X should have at least two features(predictiors)")
		#简单规则缺失值填补
		imputed_ini = X.copy() 
		if self.ini_fill:
			for i in np.arange(X.shape[1]):
				imputed_ini[:, i:i+1] = self.imputer_reg.fit_transform(X[:, i].reshape(-1,1))
		#print('fit:imputed_ini')
		#print(imputed_ini)
		#将有缺失值的特征，按缺失值个数来先后预测
		X_nan = np.isnan(X) #TODO 判断矩阵中对应位置是否是nan值的
		num_nan_desc = X_nan.sum(axis=0).argsort()[::-1] #TODO 按照缺失值的多少排序
		imputed_X = imputed_ini.copy() 		#TODO
		self.gamma_ = []  #TODO 记录每次的损失值，当损失变化不大时，则提前停止训练
		# self.estimators_ =	[xgb.XGBRegressor(**self.params) for i in range(X.shape[1])] #TODO 构造和属性数量相同的xgboost回归模型
		# TODO 构造和属性数量相同的xgboost回归模型,
		self.estimators_ =	[xgb.XGBRegressor(**self.params) if i not in self.cat_list else xgb.XGBClassifier(**self.params) for i in range(X.shape[1])]

		#对于每个属性，训练有个模型
		# for i in range(X.shape[1]):
		# 	#训练集
		# 	# trianX trainY
		# 	trainX = np.delete(imputed_X, i, 1)
		# 	trainY = imputed_X[:,i]
		# 	self.estimators_[i].fit(trainX,trainY)

		#获取各列缺失值的bool值
		self.iter_ = 0
		self.estimators_ = self.estimators_ * self.max_iter #TODO 复制多个模型
		#TODO 遍历全部的迭代次数
		for iter in np.arange(self.max_iter):  #迭代次数
			#TODO 按照缺失值的数量遍历每一列
			for i in num_nan_desc:  #TODO 缺失数据的排序,按照缺失数据较少开始排序。
				## 获取缺失值信息
				i_nan_index = X_nan[:, i]  #TODO 获取缺失数据最少的一列
				## 查看该特征是否有缺失值，如果没有缺失则跳出
				#if np.sum(i_nan_index) == 0:
				#	break
				## 删除待估计的列，构造训练集输入
				X_1 = np.delete(imputed_X, i, 1)
				## 提取估计数值非空的行，构建训练集与标签
				X_train = X_1[~i_nan_index]  # ~ 对于true false变量取反，提取特定位置的数据 TODO 此处获取估计值非空的其他属性值作为树结构的模型输入

				y_train = imputed_X[~i_nan_index, i]  # TODO 获取训练集中，估计值非空的值

				self.estimators_[iter*X.shape[1]+i].fit(X_train, y_train)  # TODO 训练模型
				#
				X_pre = X_1[i_nan_index]  # TODO 获取缺失数据的行
				if len(X_pre)>0:
					imputed_X[i_nan_index, i] = self.estimators_[int(iter*X.shape[1]+i)].predict(X_pre)  # TODO 估计缺失数据

			self.iter_ += 1  #TODO 迭代次数增加，此处迭代的目的，多次估计填充缺失值。

			# print((imputed_X-imputed_ini))
			# print(imputed_X.var(axis=0))
			# print(((imputed_X-imputed_ini)**2/(1e-6+imputed_X.var(axis=0))).sum())
			# print(1e-6+X_nan.sum())
			# #TODO 训练模型提前退出机制
			# gamma = ((imputed_X-imputed_ini)**2/(1e-6+imputed_X.var(axis=0))).sum()/(1e-6+X_nan.sum())
			# self.gamma_.append(gamma)
			# if np.abs(np.diff(self.gamma_[-2:])) < self.tol: #模型拟合直接退出
			# 	break
			#for test
			# print(imputed_X)
		print('debug')
		return self 
	def transform_one_row(self,X):
		#TODO 实时系统中完成 1005
		#判断数据是否有缺失，如果有则填充

		X = check_array(X, dtype=np.float64, force_all_finite=False)  # TODO 验证数据的类型
		if X.shape[1] == 1:  # 数据维度需要大于1
			raise ValueError("your X should have at least two features(predictiors)")
		if X.shape[0] != 1:
			raise ValueError("your X should input one row data")
		# TODO 简单规则缺失值填补
		imputed_ini = X.copy()
		for i in range(imputed_ini.shape[1]):
			if np.isnan(imputed_ini[0,i]):
				imputed_ini[0,i] = self.params_mean[i]

		# print('transform:imputed_ini')
		# print(imputed_ini)
		X_nan = np.isnan(X)  # 获取nan缺失的情况
		num_nan_desc = X_nan.sum(axis=0).argsort()[::-1]  # 按照nan数量排序属性
		# TODO 排序缺失值的数量
		for iter in np.arange(self.iter_):
			for i in num_nan_desc:
				i_nan_index = X_nan[:, i]
				if np.sum(i_nan_index) == 0:
					break
				# TODO 删除估计的列
				X_1 = np.delete(imputed_ini, i, 1)
				# TODO 提取包含缺失值的行
				X_pre = X_1[i_nan_index]
				imputed_ini[i_nan_index, i] = self.estimators_[iter * X.shape[1] + i].predict(X_pre)
				# print(imputed_ini[i_nan_index, i])
		return imputed_ini

	def transform(self, X):
		#TODO 对于整体数据进行转化, 数据集行数大于1，不存在完全为空的列
		X = check_array(X, dtype=np.float64, force_all_finite=False) # TODO 验证数据的类型
		if X.shape[1] == 1:  #数据维度需要大于1
			raise ValueError("your X should have at least two features(predictiors)")

		#TODO 简单规则缺失值填补
		imputed_ini = X.copy()
		if self.ini_fill:  #TODO 缺失数据，先简单填充均值
			for i in np.arange(X.shape[1]):
				imputed_ini[:, i:i+1] = self.imputer_reg.fit_transform(X[:, i].reshape(-1,1))

		#print('transform:imputed_ini')
		#print(imputed_ini)
		X_nan = np.isnan(X)  #获取nan缺失的情况
		num_nan_desc = X_nan.sum(axis=0).argsort()[::-1]  #按照nan数量排序属性
		#TODO 排序缺失值的数量
		for iter in np.arange(self.iter_):
			for i in num_nan_desc:
				i_nan_index = X_nan[:, i]
				if np.sum(i_nan_index) == 0:
					break
				#TODO 删除估计的列
				X_1 = np.delete(imputed_ini, i, 1)
				#TODO 提取包含缺失值的行
				X_pre = X_1[i_nan_index]
				imputed_ini[i_nan_index, i] = self.estimators_[iter*X.shape[1]+i].predict(X_pre)
		return imputed_ini
def get_score(originD,preD,subindex):
	#TODO 计算特定位置的均方根误差
	sum = 0
	for i in range(len(subindex)):
		sum += (originD[i,subindex[i]] - preD[i,subindex[i]])**2
	return np.sqrt(sum/len(subindex))
def _objective_fn(params_space):
	# TODO 本函数用于训练一个xgboost模型，
	# 使用贝叶斯模型调参。
	# 初始化模型
	# model = xgb.XGBRegressor(max_depth=params_space['max_depth'],n_estimators=int(params_space['n_estimators']),
	# 		subsample=params_space['subsample'],colsample_bytree=params_space['colsample_bytree'],
	# 		learning_rate=params_space['learning_rate'], reg_alpha=params_space['reg_alpha'])
	#TODO 数据的读取, trainX
	# for item in params_space:
	# 	print(item)
	params = {
		'max_depth': int(params_space['max_depth']),
		'n_estimators': int(params_space['n_estimators']),
		'subsample': params_space['subsample'],
		'colsample_bytree': params_space['colsample_bytree'],
		'learning_rate': params_space['learning_rate'],
		'reg_alpha': params_space['reg_alpha'],
		'gamma':params_space['gamma'],


	}
	data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
	oped_data = data.dropna()
	oped_data = oped_data.iloc[:,-6:].values
	#TODO 随机抽取若干idnex 	#TODO 获取10%的测试数据的index作为验证机
	is_train = np.array([True for i in range(oped_data.shape[0])])
	index = random.sample(range(0,oped_data.shape[0]-1), (int)(0.1*len(oped_data)))
	sub_index = [int(item%6) for item in index]  #挖出部分nan值
	for i in index:
		is_train[i] = False
	trainD = oped_data[is_train]
	# trainX = trainD[:,1:]
	# trainY = trainD[:,0]
	testD = oped_data[~is_train]
	oped_testD = oped_data[~is_train]
	for row, col in enumerate(sub_index,start=0):
		oped_testD[row,col] = np.nan
	#将testD中数据挖出部分属性
	# testX = testD[:,1:]
	# testY = testD[:,0:]
	#TODO 构造模型
	model = MultiXGboost(params)
	# TODO 训练模型,计算损失值
	model.fit(trainD)
	filledX = model.transform(oped_testD)
	score = get_score(testD,filledX,sub_index)
	print('score',score)
	return {'loss': score, 'status': STATUS_OK}


def run_params_opt():
	space = {
		'max_depth': hp.quniform("x_max_depth", 2, 20, 1),
		'n_estimators': hp.quniform("n_estimators", 100, 1000, 1),
		'subsample': hp.uniform('x_subsample', 0.7, 1),
		'colsample_bytree': hp.uniform('x_colsample_bytree', 0.1, 1),
		'learning_rate': hp.uniform('x_learning_rate', 0.01, 0.1),
		'reg_alpha': hp.uniform('x_reg_alpha', 0.1, 1),
		'gamma': hp.uniform('x_gamma',0,0.3)
	}
	#min_child_weight
	#gamma:叶节点分裂时，最小的损失函数下降值，默认为0，参数越大越保守。
	trials = Trials()
	best = fmin(fn=_objective_fn,
				space=space,
				algo=tpe.suggest, #tpe算法
				max_evals=100,
				# max_evals=1,
				trials=trials)

	print(best)
	with open('auxiliaryFile/xgboost_params.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
		print(best, file=fileHandle)
	# with open('auxiliaryFile/xgboost.yaml','w',encoding='utf-8') as f:
	# 	yaml.dump(best,f)
	# print('DEUBG')
'''
max_evals=100, 
{'n_estimators': 360, 'x_colsample_bytree': 0.7166589051315236, 'x_gamma': 0.012954988155888664, 'x_learning_rate': 0.09067065786724686, 'x_max_depth': 4, 'x_reg_alpha': 0.35325351285730894, 'x_subsample': 0.8719641998030547}
 best loss: 19.874578660559784
'''


# def read_params(path):
# 	file = open(path,'r')
# 	js = file.read()
# 	dic = json.loads(js)
# 	print(dic)
# 	file.close()
# 	return dic
if __name__ == '__main__':
	#
	run_params_opt()
	# 读取失败，格式存在问题
	# path = 'auxiliaryFile/xgboost_params.txt'
	# dic = read_params(path)






