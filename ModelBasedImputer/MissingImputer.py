#https://github.com/sjtupig/MissingImputer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer as Imputer #简单的插值方法，
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import xgboost as xgb 
import lightgbm as lgb
#TODO 训练过程对于全部缺失部分进行简单的插值补齐（均值），对于每个属性建立模型，以其他属性值为输入，该属性为目标值
# 转化阶段，同样预先填充
class MissingImputer(BaseEstimator, TransformerMixin):
	def __init__(self, max_iter = 10, ini_fill = True, ini_strategy_reg = 'mean',
		ini_strategy_clf = 'most_frequent', with_cat = False, 
		cat_index = None, tol = 1e-3, model_reg = "knn", model_clf = "knn"):
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
		self.ini_fill = ini_fill
		self.max_iter = max_iter
		self.imputer_reg = Imputer(strategy = ini_strategy_reg)  #TODO 使用均值预填充
		self.imputer_clf = Imputer(strategy = ini_strategy_clf)
		self.with_cat = with_cat
		self.cat_index = cat_index
		self.tol = tol
		self.model_reg = model_reg
		self.model_clf = model_clf
		if (not self.ini_fill) and (self.model_reg not in ('lightgbm', 'xgboost')) and (self.model_clf not in ('lightgbm', 'xgboost')):
			raise ValueError("ini_fill = False only work when prams is lightgbm or xgboost")


	def fit(self, X, y = None, model_params = {'regressor':{}, 'classifier':{}}):
		'''
		-model_params:params for models,it should be a map
		'''
		X = check_array(X, dtype=np.float64, force_all_finite=False)

		if X.shape[1] == 1: # 需要多维的数据
			raise ValueError("your X should have at least two features(predictiors)")

		#简单规则缺失值填补
		imputed_ini = X.copy() 
		if self.ini_fill:
			for i in np.arange(X.shape[1]):
				if self.with_cat and i in self.cat_index:
					imputed_ini[:, i:i+1] = self.imputer_clf.fit_transform(X[:, i].reshape(-1,1))
				else:
					imputed_ini[:, i:i+1] = self.imputer_reg.fit_transform(X[:, i].reshape(-1,1))

		#print('fit:imputed_ini')
		#print(imputed_ini)
		#将有缺失值的特征，按缺失值个数来先后预测
		X_nan = np.isnan(X) #判断矩阵中对应位置是否是nan值的
		num_nan_desc = X_nan.sum(axis=0).argsort()[::-1] #按照缺失值的多少排序

		imputed_X = imputed_ini.copy()
		self.gamma_ = []
		#set prams params 创建属性数量的模型，TODO xgboost模型为属性的数量
		# if self.model_reg == 'xgboost' and self.model_clf == 'lightgbm':
		# 	self.estimators_ = [xgb.XGBClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else lgb.LGBMRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'xgboost' and self.model_clf == 'xgboost':
			params = {'n_estimators': 360, 'colsample_bytree': 0.7166589051315236, 'gamma': 0.012954988155888664,
					  'learning_rate': 0.09067065786724686, 'max_depth': 4, 'reg_alpha': 0.35325351285730894,
					  'subsample': 0.8719641998030547}

			self.estimators_ = [xgb.XGBClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else xgb.XGBRegressor(**params) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'xgboost' and self.model_clf == 'randomforest':
		# 	self.estimators_ = [xgb.XGBClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else RandomForestRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'xgboost' and self.model_clf == 'knn':
		# 	self.estimators_ = [xgb.XGBClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else KNeighborsRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'lightgbm' and self.model_clf == 'lightgbm':
			self.estimators_ = [lgb.LGBMClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else lgb.LGBMRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'lightgbm' and self.model_clf == 'xgboost':
		# 	self.estimators_ = [lgb.LGBMClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else xgb.XGBRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'lightgbm' and self.model_clf == 'randomforest':
		# 	self.estimators_ = [lgb.LGBMClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else RandomForestRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'lightgbm' and self.model_clf == 'knn':
		# 	self.estimators_ = [lgb.LGBMClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else KNeighborsRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'randomforest' and self.model_clf == 'lightgbm':
		# 	self.estimators_ = [RandomForestClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else lgb.LGBMRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'randomforest' and self.model_clf == 'xgboost':
		# 	self.estimators_ = [RandomForestClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else xgb.XGBRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'randomforest' and self.model_clf == 'randomforest':
			self.estimators_ = [RandomForestClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else RandomForestRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'randomforest' and self.model_clf == 'knn':
		# 	self.estimators_ = [RandomForestClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else KNeighborsRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'knn' and self.model_clf == 'lightgbm':
		# 	self.estimators_ = [KNeighborsClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else lgb.LGBMRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'knn' and self.model_clf == 'xgboost':
		# 	self.estimators_ = [KNeighborsClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else xgb.XGBRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		# if self.model_reg == 'knn' and self.model_clf == 'randomforest':
		# 	self.estimators_ = [KNeighborsClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else RandomForestRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'knn' and self.model_clf == 'knn':
			self.estimators_ = [KNeighborsClassifier(**model_params['classifier']) if self.with_cat and i in self.scat_index else KNeighborsRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]

		#获取各列缺失值的bool值
		self.iter_ = 0
		self.estimators_ = self.estimators_ * self.max_iter #TODO 每个属性建立一个模型，并且每个模型有多棵树
		#TODO 遍历全部的迭代次数
		for iter in np.arange(self.max_iter):  #迭代次数
			#TODO 按照缺失值的数量遍历每一列
			for i in num_nan_desc:  #TODO 缺失数据的排序, 先对缺失值较少的数据进行填充，然后用填充好的数据进行，其他属性的估计
				#获取缺失值信息
				i_nan_index = X_nan[:, i] # 获取缺失数据最少的一列
				#查看该特征是否有缺失值，如果没有缺失则跳出
				if np.sum(i_nan_index) == 0:
					break
				#删除待估计的列，构造训练集输入
				X_1 = np.delete(imputed_X, i, 1)
				#提取估计数值非空的行，构建训练集与标签
				X_train = X_1[~i_nan_index]  #~ 对于true false变量取反，提取特定位置的数据 TODO 此处获取估计值非空的其他属性值作为树结构的模型输入

				y_train = imputed_X[~i_nan_index, i] #TODO 获取训练集中，估计值非空的值

				X_pre = X_1[i_nan_index]  #TODO 获取缺失数据的行
				self.estimators_[iter*X.shape[1]+i].fit(X_train, y_train) #TODO 训练模型 第0，6，12，18 ...表示为用一属性的模型

				imputed_X[i_nan_index, i] = self.estimators_[iter*X.shape[1]+i].predict(X_pre)  #TODO 估计缺失数据

			self.iter_ += 1

			gamma = ((imputed_X-imputed_ini)**2/(1e-6+imputed_X.var(axis=0))).sum()/(1e-6+X_nan.sum())
			self.gamma_.append(gamma)
			if np.abs(np.diff(self.gamma_[-2:])) < self.tol: #模型拟合直接退出
				break
		#for test
		# print(imputed_X)

		return self 

	def transform(self, X):
		X = check_array(X, dtype=np.float64, force_all_finite=False) # TODO 验证数据的类型
		if X.shape[1] == 1:  #数据维度需要大于1
			raise ValueError("your X should have at least two features(predictiors)")

		#简单规则缺失值填补
		imputed_ini = X.copy() 
		if self.ini_fill:  #TODO 缺失数据，先简单填充均值
			for i in np.arange(X.shape[1]):
				if self.with_cat and i in self.cat_index:
					imputed_ini[:, i:i+1] = self.imputer_clf.fit_transform(X[:, i].reshape(-1,1))
				else:
					imputed_ini[:, i:i+1] = self.imputer_reg.fit_transform(X[:, i].reshape(-1,1))

		#print('transform:imputed_ini')
		#print(imputed_ini)
		X_nan = np.isnan(X)  #获取nan缺失的情况
		num_nan_desc = X_nan.sum(axis=0).argsort()[::-1]  #按照nan数量排序属性

		for iter in np.arange(self.iter_):
			for i in num_nan_desc:
				i_nan_index = X_nan[:, i]
				if np.sum(i_nan_index) == 0:
					break

				X_1 = np.delete(imputed_ini, i, 1)
				X_pre = X_1[i_nan_index]

				imputed_ini[i_nan_index, i] = self.estimators_[iter*X.shape[1]+i].predict(X_pre)


		'''for i, estimators in enumerate(self.estimators_):
			i_nan_index = X_nan[:, i]
			if np.sum(i_nan_index) == 0:	
				continue

			X_1 = np.delete(imputed_ini, i, 1)
			X_pre = X_1[i_nan_index]
			X[i_nan_index, i] = estimators.predict(X_pre)'''

		return imputed_ini




