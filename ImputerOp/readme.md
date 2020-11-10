#文件夹内文件介绍 及 运行顺序
# 可执行文件
## 1: .ipynb文件为模块测试使用文件

## 2：compareImputingMethods.py 
### 2.1 定义daraframe格式数据缺失统计函数* check_loss(df,item,outfile) 
Line 35: for row in subdf.index[0:2000]: #TODO 前三个月用于测试插值方法，真实使用去除区间
### 2.2 对比插值方法。
global mean; 
local mean; 
insert_data_bymethod(*):xgboost,lightgbm,knn,randomforest;
my_multi_viewer_estimate

## 3:RunotherMethods.py
### 3.1 run_test(item,year,outfile)
该函数使用2.2中的多种方法对于测试数据进行插值

## 4：multiobjXGBoost.py
### 4.1: MultiXGboost() 
定义自己的xgboost类，实现对于多属性插值
### 4.2：run_params_opt() 
使用贝叶斯优化寻参算法查找xgboost模型的
{100轮的贝叶斯优化估计，所得参数}
{'n_estimators': 360.0, 'x_colsample_bytree': 0.7166589051315236, 'x_gamma': 
0.012954988155888664, 'x_learning_rate': 0.09067065786724686, 'x_max_depth': 4.0, 
'x_reg_alpha': 0.35325351285730894, 'x_subsample': 0.8719641998030547}

## 5：IMA.py
### 5.1 IMA（）
定义IMA插值方法类，类中记录插值方法所需要的数据，表，及模型参数等。
### 5.2 run_test()
用于测试插值效果，与3中的方法形成对比。

###6：muViewer.py
本文件定义了multiviewer插值方法以及测试代码

#文件夹
##1 auxiliaryFile
该文件夹记录程序运行过程中产生的变量（如处理的中间插值数据，以及模型的参数），目的为加速调试及运行的速度

##2 origin
本文件夹记录代码运行前所需的文件(如测试数据信息checkloss*.csv，皮尔森邻居信息等)，用于插值验证过程

## result
本文件夹记录程序的执行结果(output_result.txt)

## logfile
本文件夹备份工程历史文件，


## 

# 代码运行顺序
Runothermethod.run_test(item,year,outfile) :输出对比方法的查插值结果，该代码还会输出缺失统计文件用于后续实验 ->  
multiobjXGBoost.run_params_opt()  ：gxboost参数寻优 ->
IMA.run_test() : 运行IMA算法代码，输出结果 ->
muViewer.run_mu_viewer():运行mu_viewer,修改mu_viwer定义knn_val的大小（-1，7），修改邻居的数量。




