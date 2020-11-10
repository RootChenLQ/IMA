# import os,sys
# sys.path.append("path")
from dateutil.relativedelta import relativedelta
from utils.func import getMAE, getMRE, compute_score_byInterval
from ModelBasedImputer.MissingImputer import MissingImputer
# from util.multi_view_impute_fn import get_dataMetrix,\
#     get_SimpleExponentialSmooth,get_InverseDistanceWeighting, \
#     get_correlation_of_two_array, get_UserBasedCollaborativeFiltering,\
#     get_ItembasedCollaborativefiltering, getweightedNanMean,getweightedNanMean2,\
#     get_nan_rate
from utils.multi_view_impute_fn import *
from pathlib import Path
#TODO sklearn 模型存储与加载
import joblib
def check_loss(df,item,outfile):
    #TODO 本函数用于记录PM2.5缺失值，用于比较插值方法
    #return data loss index 插值方法验证时，删除特定位置的值，用于真值验证
    #创建id类，划分不同的station_id
    #查找PM2.5缺失的位置，记录下一个月的缺失值位置
    #记录下一月，未缺失数据的位置。

    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df_c = df_clip(df,'station_id')
    first_out = True
    for subdf in df_c.list:
        # i = 0
        # item = 'PM2.5'
        record = pd.DataFrame(columns=['station_id', 'row']) # record each station_id
        count = 0
        size = subdf.index.max()
        max_date = subdf['utc_time'].max()
        id_ = subdf.loc[0,'station_id']
        nan_list = subdf[item].isna()
        #print(nan_list)
        for row in subdf.index: #TODO 前三个月
            #判断该行item值是否为空
            # if np.isnan(df_c.list[i].loc[row, item]):
            if nan_list[row]:
                time = subdf.loc[row,'utc_time'] #获取时间
                #print(time)
                time = time + relativedelta(months=+1) #计算下一个月时间
                if time > max_date: #如果时间查过最大值，则无需查找
                    break
                #判断缺失数据是否存在数据表中
                # time = pd.to_datetime(time)
                # print(df_c.list[i]['utc_time'])
                for next_row in range(row, size): #往后查时间戳是否存在
                    if  subdf.loc[next_row,'utc_time'] == time: #判断时间戳是否存在
                        #print(subdf.loc[next_row,:] )
                        # if not np.isnan(df_c.list[i].loc[next_row, item]):#判断下一个月的数据是否存在
                        if not nan_list.loc[next_row]:
                            #print(time, item, 'is not nan')
                            count += 1
                            # s_ = pd.Series([id_,next_row])
                            orig_data = subdf.loc[next_row, item]
                            df_ =  pd.DataFrame([[id_,next_row,orig_data]],columns=['station_id','row','data'])  #
                            record = pd.concat([record,df_], axis=0)
                        # print(t)
                        break #stop searching
            else: #该行存在数据
                pass
        if first_out:
            first_out = False
            record.to_csv(outfile, index = False)
        else:
            record.to_csv(outfile, index = False, header=False, mode='a')
        print('test data size=', count)
        # if time in df_c.list[0]['utc_time']:
        #     row = df_c.list[0][df_c.list[0]['utc_time']==time].index[0]
        #     print('time exit..')
        #     if not np.isnan(df_c.list[0].loc[row,'PM2.5']):
        #         print(time,'PM2.5','is not nan')
        # print(time)
        # print(row)

    # print(df.head())
def del_data_by_index(data,indexdf,item,outfile):
    #删除对应index上的数据
    df = data.copy()  # avoid the value chaged by nan
    begin = indexdf.index[0]
    for row in indexdf.index:  # station_id, index 将realY填充到indexdf文件中
        if row % 1000 == 0:
            print(row)
        station = indexdf.loc[row, 'station_id']  # 获取station_id
        check_row = indexdf.loc[row, 'row']  # 获取行号
        real_row = df[(df['station_id'] == station) & (df['Unnamed: 0'] == begin + check_row)].index[0]  # 获取真实的数据行
        df.loc[real_row, item] = np.NaN  # 固定位置置空
    file = 'origin/del_testdata_'+outfile
    df.to_csv(file,index=False)
    return df
def is_array_equal(arr1,arr2):
    if len(arr1) != len(arr2):
        return False
    else:
        for i in range(len(arr1)):
            if arr1[i]!=arr2[i]:
                return False
        return True
def arrIn(arr1,arr2):
    # TODO 判断第一个数组中的元素是否全部存在在第二个数组中
    if  len(arr1)>len(arr2):
        return False
    else:
        arrin_ = True
        for val in arr1:
            if val not in arr2:
                arrin_=False
                break
        return arrin_

# mae mean absolute error
# mre mean relative error
def insert_global_mean(df,indexdf,item,outfile,interval):
    # check_list =
    df_c = df_clip(df, 'station_id')

    mean = df[item].mean()
    # print(mean)
    index_c = df_clip(indexdf,'station_id')
    # assert is_array_equal(df_c.id,index_c.id), "station sizes are not equal"
    # assert arrIn(index_c.id,df_c.id),'Vales in param A is not all in  param A'
    size = len(index_c.list)
    realY = []
    estiY = []
    for i in range(size): #search each station id
        #获取index列表的长度，按照station_id取值
        subindex = index_c.list[i]
        temp_index_ = subindex.index.values[0]
        print(subindex.loc[temp_index_,'station_id'])
        list_index = df_c.get_listIndex_by_id(subindex.loc[temp_index_,'station_id'])
        subdf = df_c.list[list_index]
        #
        #mean = np.nanmean(subdf['PM2.5'].values)#insert mean
        # print(np.nanmean(subdf['PM2.5'].values))
        for row in subindex.index:
            index_ = subindex.loc[row,'row']
            realY.append(subdf.loc[index_,item])
            estiY.append(mean)
    indexdf['realY'] = pd.Series(realY)
    indexdf['esti_data'] = pd.Series(estiY)
    file = 'result/globalMean'+outfile
    indexdf.to_csv(file)
    #写结果到txt中
    # mae = getMAE(realY,estiY)
    # mre = getMRE(realY,estiY)
    # with open('result/output_result.txt', mode='a') as fileHandle: #'w'写，'r'读，'a'添加
    #     print('global_mean:',' ','MAE:',mae,' ','MRE:',mre, file=fileHandle)
    length = len(df_c.list[0])
    compute_score_byInterval(indexdf, item, 'Global Mean', length, interval)
    # return mae, mre

def insert_history_mean(df, oped_df, indexdf, item, w, outfile,interval):
    #TODO 修改填充机制，删除带填充部分
    #再填充，如果数据完全缺失则，使用全局均值来填充
    globalmean = oped_df[item].mean()
    realY = []
    estiY = []
    df_c = df_clip(oped_df, 'station_id') #已经删除测试位置的数据
    df_copy_c = df_clip(df, 'station_id') #原始数据
    index_c = df_clip(indexdf, 'station_id') #被删除的位置索引
    # assert is_array_equal(df_c.id, index_c.id), "station sizes are not equal"
    # assert is_array_equal(df_c.id, df_copy_c.id), "station sizes are not equal"
    size = len(index_c.list)  #station_id的数量
    for i in range(size):  # search each station id
        #subdf = df_c.list[i]
        #subdf_origin = df_copy_c.list[i]
        subindex = index_c.list[i]
        temp_index_ = subindex.index.values[0]
        list_index = df_c.get_listIndex_by_id(subindex.loc[temp_index_, 'station_id'])
        subdf = df_c.list[list_index]
        list_index = df_copy_c.get_listIndex_by_id(subindex.loc[temp_index_, 'station_id'])
        subdf_origin = df_copy_c.list[list_index]
        # mean = np.nanmean(subdf['PM2.5'].values)#insert mean
        # print(np.nanmean(subdf['PM2.5'].values))
        for row in subindex.index:
            index_ = subindex.loc[row, 'row'] #获取行号
            realY.append(subdf_origin.loc[index_, item]) #添加原始的数据值
            #获取当前row,前后12个数据的平均值
            if pd.isna(subdf[item][index_-(int)(w/2): index_+(int)(w/2 +1)]).all(): #
                mean = globalmean
            else:
                mean = subdf[item][index_-(int)(w/2): index_+(int)(w/2 +1)].mean() #计算删除检测值后的数据均值
            # print(subdf[item][index_-6:index_+6])
            estiY.append(mean)
    indexdf['realY'] = pd.Series(realY)
    indexdf['esti_data'] = pd.Series(estiY)
    file = 'result/historyMean' + outfile
    indexdf.to_csv(file)
    #写结果到txt中
    #
    length = len(df_c.list[0])
    compute_score_byInterval(indexdf, item, 'Local Mean', length, interval)
# def insert_xgboost(df, indexdf, item,outfile):
#     #读取原始数据
#     #记录历史数据，挖空缺失数据
#     # fileHandle = open('log.txt', mode='w')
#     df_copy = df.copy()  #avoid the value chaged by nan
#     # print(df.loc[4306,:])
#     indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))
#     for row in indexdf.index:  #station_id, index
#         station = indexdf.loc[row,'station_id'] #获取station_id
#         check_row =  indexdf.loc[row,'row']  #获取行号
#         real_row = df[(df['station_id'] == station) & (df['Unnamed: 0'] == check_row)].index[0] #获取真实的数据行
#         indexdf.loc[row,'realY'] = df_copy.loc[real_row,'PM2.5']  #填充缺失值
#         df.loc[real_row,'PM2.5'] = np.NaN  #固定位置置空
#     numerialMatric = df.iloc[:,3:] #获取数据区域
#     position = 0  #根据cloumn名称缺失列号
#     for j in range(numerialMatric.columns.size):
#         if numerialMatric.columns.values[j] == item:
#             position = j  #获取item所在的位置
#             break
#     Imputer = MissingImputer(ini_fill=True, model_reg="xgboost", model_clf="xgboost") #定义xgboost插入方法
#     imputed_data = Imputer.fit(numerialMatric).transform(numerialMatric.copy())
#     #查找插入值
#     for row in indexdf.index:  #station_id, index
#         station = indexdf.loc[row,'station_id']
#         check_row =  indexdf.loc[row,'row']
#         # print(station, check_row)
#         #获取index值
#         # print('before insert',indexdf.loc[row,:], file=fileHandle)
#         real_row = df[(df['station_id'] == station) & (df['Unnamed: 0'] == check_row)].index.tolist()[0]
#         indexdf.loc[row,'estiY'] = imputed_data[real_row,position]
#         # print('after insert',indexdf.loc[row, :], file=fileHandle)
#     file = 'xgboost' + outfile
#     indexdf.to_csv(file, index = False)
#     mae = getMAE(indexdf['realY'].tolist(), indexdf['estiY'].tolist())
#     mre = getMRE(indexdf['realY'].tolist(), indexdf['estiY'].tolist())
#     with open('result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#        print('xgboost', ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     # print('xgboost', ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     # fileHandle.close()
#     return mae, mre
# def insert_lightbgm(df, indexdf, item,outfile):
#     #读取原始数据
#     #记录历史数据，挖空缺失数据
#     # fileHandle = open('log.txt', mode='w')
#     df_copy = df.copy()  #avoid the value chaged by nan
#     # print(df.loc[4306,:])
#     indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))
#     for row in indexdf.index:  #station_id, index
#         station = indexdf.loc[row,'station_id'] #获取station_id
#         check_row =  indexdf.loc[row,'row']  #获取行号
#         real_row = df[(df['station_id'] == station) & (df['Unnamed: 0'] == check_row)].index[0] #获取真实的数据行
#         indexdf.loc[row,'realY'] = df_copy.loc[real_row,'PM2.5']  #填充缺失值
#         df.loc[real_row,'PM2.5'] = np.NaN  #固定位置置空
#     numerialMatric = df.iloc[:,3:] #获取数据区域
#     position = 0  #根据cloumn名称缺失列号
#     for j in range(numerialMatric.columns.size):
#         if numerialMatric.columns.values[j] == item:
#             position = j  #获取item所在的位置
#             break
#     Imputer = MissingImputer(ini_fill=True, model_reg="lightgbm", model_clf="lightgbm") #定义xgboost插入方法
#     imputed_data = Imputer.fit(numerialMatric).transform(numerialMatric.copy())
#     #查找插入值
#     for row in indexdf.index:  #station_id, index
#         station = indexdf.loc[row,'station_id']
#         check_row =  indexdf.loc[row,'row']
#         # print(station, check_row)
#         #获取index值
#         # print('before insert',indexdf.loc[row,:], file=fileHandle)
#         real_row = df[(df['station_id'] == station) & (df['Unnamed: 0'] == check_row)].index.tolist()[0]
#         indexdf.loc[row,'estiY'] = imputed_data[real_row,position]
#         # print('after insert',indexdf.loc[row, :], file=fileHandle)
#     file = 'xgboost' + outfile
#     indexdf.to_csv(file, index = False)
#     mae = getMAE(indexdf['realY'].tolist(), indexdf['estiY'].tolist())
#     mre = getMRE(indexdf['realY'].tolist(), indexdf['estiY'].tolist())
#     with open('result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#        print('lightgbm', ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     # print('xgboost', ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     # fileHandle.close()
#     return mae, mre
# def insert_knn(df,indexdf,item,outfile):
#     # 读取原始数据
#     # 记录历史数据，挖空缺失数据
#     # fileHandle = open('log.txt', mode='w')
#     df_copy = df.copy()  # avoid the value chaged by nan
#     # print(df.loc[4306,:])
#     indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))
#     for row in indexdf.index:  # station_id, index
#         station = indexdf.loc[row, 'station_id']  # 获取station_id
#         check_row = indexdf.loc[row, 'row']  # 获取行号
#         real_row = df[(df['station_id'] == station) & (df['Unnamed: 0'] == check_row)].index[0]  # 获取真实的数据行
#         indexdf.loc[row, 'realY'] = df_copy.loc[real_row, 'PM2.5']  # 填充缺失值
#         df.loc[real_row, 'PM2.5'] = np.NaN  # 固定位置置空
#     numerialMatric = df.iloc[:, 3:]  # 获取数据区域
#     position = 0  # 根据cloumn名称缺失列号
#     for j in range(numerialMatric.columns.size):
#         if numerialMatric.columns.values[j] == item:
#             position = j  # 获取item所在的位置
#             break
#     Imputer = MissingImputer(ini_fill=True, model_reg="knn", model_clf="knn")  # 定义xgboost插入方法
#     imputed_data = Imputer.fit(numerialMatric).transform(numerialMatric.copy())
#     # 查找插入值
#     for row in indexdf.index:  # station_id, index
#         station = indexdf.loc[row, 'station_id']
#         check_row = indexdf.loc[row, 'row']
#         # print(station, check_row)
#         # 获取index值
#         # print('before insert',indexdf.loc[row,:], file=fileHandle)
#         real_row = df[(df['station_id'] == station) & (df['Unnamed: 0'] == check_row)].index.tolist()[0]
#         indexdf.loc[row, 'estiY'] = imputed_data[real_row, position]
#         # print('after insert',indexdf.loc[row, :], file=fileHandle)
#     file = 'knn' + outfile
#     indexdf.to_csv(file, index=False)
#     mae = getMAE(indexdf['realY'].tolist(), indexdf['estiY'].tolist())
#     mre = getMRE(indexdf['realY'].tolist(), indexdf['estiY'].tolist())
#     with open('result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#         print('knn', ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     # print('xgboost', ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     # fileHandle.close()
#     return mae, mre
def insert_data_bymethod(df,oped_data,indexdf,item,outfile,method,interval):
    #TODO
    # https://github.com/sjtupig/MissingImputer 插值的方法
    # df 原始数据; oped_data 删除对应index上的数据;index_df 被删除位置的station_id 和index;item 被删除的列名
    # outfile输出文件名;method 方法
    # 读取原始数据
    # 记录历史数据，挖空缺失数据
    # fileHandle = open('log.txt', mode='w')
    assert method in ['xgboost','lightgbm','randomforest','knn'] , "Choose method from ['xgboost','lightgbm','randomforest','knn']"
    # df_copy = df.copy()  # avoid the value chaged by nan
    # print(df.loc[4306,:])
    indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))  #真实数据
    indexdf['esti_data'] = pd.Series(np.zeros(len(indexdf)))  #估计数据


    numerialMatric = oped_data.iloc[:, -6:]  # 获取删除测试数据后的数据区域
    # numerialMatric_ori = df.iloc[:,3:]  #获得原始数据区域
    position = 0  # 根据cloumn名称缺失列号
    for j in range(numerialMatric.columns.size):
        if numerialMatric.columns.values[j] == item:
            position = j  # 获取item所在的位置
            break
    model_save = 'auxiliaryFile/'+outfile+item+method+'model.dat'
    path = Path(model_save)
    if path.exists():
        Imputer = joblib.load(path)
    else:
        Imputer = MissingImputer(max_iter = 100,ini_fill=True, model_reg=method, model_clf=method)  # 定义xgboost插入方法
        Imputer.fit(numerialMatric)
        joblib.dump(Imputer, path)
    imputed_data = Imputer.transform(numerialMatric.copy())
    df_ = pd.DataFrame(imputed_data)
    df_.to_csv('xgboost1.csv')
    # 查找插入值
    for row in indexdf.index:  # station_id, index
        if row % 5000 == 0:
            print(row)
        station = indexdf.loc[row, 'station_id']
        check_row = indexdf.loc[row, 'row']
        # print(station, check_row)
        # 获取index值
        # print('before insert',indexdf.loc[row,:], file=fileHandle)
        real_row = oped_data[(oped_data['station_id'] == station) & (oped_data['Unnamed: 0'] == check_row)].index.tolist()[0]
        indexdf.loc[row, 'esti_data'] = imputed_data[real_row, position]
        indexdf.loc[row,'realY'] = df.loc[real_row, item] #获取df原始数据

        # print('after insert',indexdf.loc[row, :], file=fileHandle)
    file = 'result/' + method + outfile
    indexdf.to_csv(file, index=False)
    # mae = getMAE(indexdf['realY'].tolist(), indexdf['estiY'].tolist())
    # mre = getMRE(indexdf['realY'].tolist(), indexdf['estiY'].tolist())
    # with open('result/output_result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
    #     print(method, ': ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
    # # print('xgboost', ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
    # # fileHandle.close()
    # return mae, mre
    length = int(oped_data['Unnamed: 0'].max())
    compute_score_byInterval(indexdf, item, method, length, interval)
# def insert_multi_viewer_estimate(oped_data, indexdf, item, outfile, neighbordf, neighbor_dis_df, knnVal=-1, wind=7):
#     #遍历index，中station_id 无训练模型
#     # indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))  #真实值已经存在
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))  #估计值
#     middle = (int)(wind / 2)  # get_SimpleExponentialSmooth window middle 时间窗口中间值
#     if knnVal == -1:
#         k = len(neighbordf)  #全部邻域信息
#     else:
#         k = knnVal           #K近邻，K紧邻信息
#     mean_ = oped_data[item].mean()  #全局平均值，当插值方法失效时，使用全局均值插入。
#     ##TODO  operation1 record real data, and clean original data 调试可注释。改用读取o'p
#     df_c = df_clip(oped_data, 'station_id')  #数据按照station_id划分
#
#     columns = 0   #列号
#     for i in range(df_c.list[0].columns.size): #获取item对应的列号
#         if df_c.list[0].columns.values[i] == item:
#             columns = i
#             break
#
#     for row in indexdf.index.values:  #逐行便利indexdf
#         # print(row)
#         if row % 1000 == 0:
#             print(row)
#         #获取stationid，row号
#         station_id = indexdf.loc[row,'station_id']  #station_id号
#
#         query_row = indexdf.loc[row,'row']  #查询行号
#         dataMatrix,neighborlist = get_dataMetrix(df_c, neighbordf, station_id, columns, query_row, w=wind, k=k) #获得邻域矩阵
#         dis_all = neighbor_dis_df.loc[station_id, :]  # get station_id neighbor distance
#         dis = dis_all[neighborlist]
#         # print(dataMatrix)
#         # print(matrix)
#         # 1
#         ses_d,ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  #获得指数平滑值
#         # print('ses_d', ses_d)
#         # get dis array
#         # 2
#         idw_d,idw_d_con = get_InverseDistanceWeighting(dataMatrix[middle, :], dis)  #
#         # print('idw_d', idw_d)
#         # 3
#         ubcf_d,ubcf_d_con = get_UserBasedCollaborativeFiltering(dataMatrix)
#         # print('ubcf_d', ubcf_d)
#         # 4
#         ibcf_d,ibcf_d_con = get_ItembasedCollaborativefiltering(dataMatrix)
#         # print('ibcf_d', ibcf_d)
#         if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):
#             # print(row, column, 'insert all NAN, we replace this by mean')
#             mean_d = mean_
#             imputed_d = mean_d
#         else:
#             list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
#             imputed_d = getweightedNanMean2(list_)
#
#         indexdf.loc[row,'estiY'] = imputed_d
#
#     str_ = str(k) +'neighbors'+'multiviewer' + 'window'+str(wind)
#     file = str_ + outfile
#     indexdf.to_csv(file)
#     # 写结果到txt中
#     mae = getMAE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     mre = getMRE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     with open('result/output_result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#         print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     return mae, mre
#
# def insert_multi_viewer_estimate_bk(oped_data, indexdf, item, outfile, neighbordf, neighbor_dis_df, knnVal=-1, wind=7):
#     #遍历index，中station_id
#     # indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))  #真实值已经存在
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))  #估计值
#     middle = (int)(wind / 2)  # get_SimpleExponentialSmooth window middle 时间窗口中间值
#     if knnVal == -1:
#         k = len(neighbordf)  #全部邻域信息
#     else:
#         k = knnVal           #K近邻，K紧邻信息
#     mean_ = oped_data[item].mean()  #全局平均值，当插值方法失效时，使用全局均值插入。
#     ##TODO  operation1 record real data, and clean original data 调试可注释。改用读取o'p
#     df_c = df_clip(oped_data, 'station_id')  #数据按照station_id划分
#
#     columns = 0   #列号
#     for i in range(df_c.list[0].columns.size): #获取item对应的列号
#         if df_c.list[0].columns.values[i] == item:
#             columns = i
#             break
#
#     for row in indexdf.index.values:  #逐行便利indexdf
#         # print(row)
#         if row % 1000 == 0:
#             print(row)
#         #获取stationid，row号
#         station_id = indexdf.loc[row,'station_id']  #station_id号
#
#         query_row = indexdf.loc[row,'row']  #查询行号
#         dataMatrix,neighborlist = get_dataMetrix(df_c, neighbordf, station_id, columns, query_row, w=wind, k=k) #获得邻域矩阵
#         dis_all = neighbor_dis_df.loc[station_id, :]  # get station_id neighbor distance
#         dis = dis_all[neighborlist]
#         # print(dataMatrix)
#         # print(matrix)
#         # 1
#         ses_d,ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  #获得指数平滑值
#         # print('ses_d', ses_d)
#         # get dis array
#         # 2
#         idw_d,idw_d_con = get_InverseDistanceWeighting(dataMatrix[middle, :], dis)  #
#         # print('idw_d', idw_d)
#         # 3
#         ubcf_d,ubcf_d_con = get_UserBasedCollaborativeFiltering(dataMatrix)
#         # print('ubcf_d', ubcf_d)
#         # 4
#         ibcf_d,ibcf_d_con = get_ItembasedCollaborativefiltering(dataMatrix)
#         # print('ibcf_d', ibcf_d)
#         if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):
#             # print(row, column, 'insert all NAN, we replace this by mean')
#             mean_d = mean_
#             imputed_d = mean_d
#         else:
#             list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
#             imputed_d = getweightedNanMean2(list_)
#
#         indexdf.loc[row,'estiY'] = imputed_d
#
#     str_ = str(k) +'neighbors'+'multiviewer' + 'window'+str(wind)
#     file = str_ + outfile
#     indexdf.to_csv(file)
#     # 写结果到txt中
#     mae = getMAE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     mre = getMRE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     with open('result/output_result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#         print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     return mae, mre

def get_pearson_neighbor(df,dis_df,item, year):  #2017-2018
    #计算df中各属性列的某一列的pearson相关系数
    #获得matrix矩阵，每一列
    #读取aq_aq_dis
    assert item in df.columns.values, "item is not in the dataframe columns"
    df_c = df_clip(df, 'station_id')
    matrix = np.zeros([len(df_c.list[0]),len(df_c.id)])
    for i, id in enumerate(df_c.id,start=0):
        # print(i, id)
        matrix[:,i] = df_c.list[i][item].to_list()
    #计算pearson值
    matrix_df = pd.DataFrame(matrix, columns = df_c.id)
    pearson_matric = matrix_df.corr(method = 'pearson')

    #排列pearson相关系数矩阵的大小，生成dis矩阵
    index_ = pearson_matric.index  # 行的index
    columns_ = pearson_matric.columns  # 列名
    # 获取数据数组
    data_array = pearson_matric.values  # 获取matrix的数值
    # print(data_array)
    # print(data_array.shape)
    aq_aq_dis_df = pearson_matric.reset_index(drop=True)
    # print(aq_aq_dis_df.head())
    new_columns = []
    for i in range(len(columns_)):  # 对df中全部行进行距离排序
        column_name = 'neighbor' + str(i+1) #设置列名 neighbor0表示自身
        new_columns.append(column_name)  # 保存新的列名
        aq_aq_dis_df[column_name] = data_array.argmax(axis=1)  # 获得最大值的位置
        data_array[aq_aq_dis_df.index, np.argmax(data_array, axis=1)] = -1e10  # 给最大值置小值
    aq_aq_neighbor = aq_aq_dis_df[new_columns].copy()  # 提取插入的neighbor数据
    # 将内容存储为aq
    for i in range(len(columns_)):
        # print(i)
        aq_aq_neighbor = aq_aq_neighbor.replace(i, columns_[i])  # 将排列信息，转化为气象站的名字
    # print(aq_aq_neighbor.head())
    # 将index 设置为aq
    person_neighbor = aq_aq_neighbor.set_index(index_)
    # print(aq_aq_neighbor.head())
    file = year + item + 'pearson_neighbor.csv'
    person_neighbor.to_csv(file)
    neighbor_df = person_neighbor
    #计算pearson 邻域矩阵距离
    neighbor_dis = neighbor_df.copy()  #复制neighbor_df数据格式
    #dis_df距离查找表
    for row_index, row_id in enumerate(neighbor_df.index.values,start=0):
        for columns_index ,column_id in enumerate(neighbor_df.loc[row_id,:],start=0):
            dis =  dis_df.loc[row_id, column_id]
            neighbor_dis.iloc[row_index, columns_index] = dis
            #print(neighbor_dis.iloc[row_index, columns_index])
    file = year + item + 'pearson_neighbor_dis.csv'
    neighbor_dis.to_csv(file)
    return neighbor_df, neighbor_dis
# def insert_pearsonBased_multi_viewer_estimate(df,indexdf,item,outfile,neighbordf,neighbor_dis_df,knnVal=5,wind=13):
#     #df已经删除测试位置的df
#     # 遍历index，中station_id
#     # check_list =
#     # print(df.loc[4306,:])
#     #indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['localMean'] = pd.Series(np.zeros(len(indexdf)))  # 真实值
#     indexdf['SES'] = pd.Series(np.zeros(len(indexdf)))  # 估计值
#     indexdf['IDW'] = pd.Series(np.zeros(len(indexdf)))  # 真实值
#     indexdf['ubcf'] = pd.Series(np.zeros(len(indexdf)))  # 估计值
#     indexdf['ibcf'] = pd.Series(np.zeros(len(indexdf)))  # 真实值
#     middle = (int)(wind / 2)  # get_SimpleExponentialSmooth window middle
#     if knnVal == -1:
#         k = len(neighbordf)
#     else:
#         k = knnVal
#     globalmean_ = df[item].mean()
#
#
#     df_c = df_clip(df, 'station_id')  # 数据按照station_id划分
#     # mean = df['PM2.5'].mean()
#     # print(mean)
#     # index_c = df_clip(indexdf, 'station_id')  #index_c也
#     # assert is_array_equal(df_c.id, index_c.id), "station sizes are not equal"
#     # size = len(df_c.list)
#     # realY = []
#     # estiY = []
#     columns = 0
#     for i in range(df_c.list[0].columns.size):  # 获取item对应的列号
#         if df_c.list[0].columns.values[i] == item:
#             columns = i
#             break
#
#     for row in indexdf.index.values:  # 逐行便利indexdf
#         if row % 1000 == 0:
#             print(row)
#         # 获取stationid，row号
#         station_id = indexdf.loc[row, 'station_id']
#         # dis = neighbor_dis_df.loc[station_id, :].values[:k]  # get station_id neighbor distance
#         query_row = indexdf.loc[row, 'row']
#         dataMatrix, neighborlist = get_dataMetrix(df_c, neighbordf, station_id, columns, query_row, w=wind, k=k)
#         dis_all = neighbor_dis_df.loc[station_id, :]  # get station_id neighbor distance
#         dis = dis_all[neighborlist]
#         # print(dataMatrix)
#         # print(matrix)
#
#         # 1
#         ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  # 获得指数平滑值
#         # print('ses_d', ses_d)
#         # get dis array
#         # 2
#         idw_d, idw_d_con = get_InverseDistanceWeighting(dataMatrix[middle, :], dis)  #
#         # print('idw_d', idw_d)
#         # 3
#         ubcf_d, ubcf_d_con = get_UserBasedCollaborativeFiltering(dataMatrix)
#         # print('ubcf_d', ubcf_d)
#         # 4
#         ibcf_d, ibcf_d_con = get_ItembasedCollaborativefiltering(dataMatrix)
#
#         if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):
#             # print(row, column, 'insert all NAN, we replace this by mean')
#             #获取id号对应的list号
#             #寻找到近期的数据数组
#             list_index = df_c.get_listIndex_by_id(station_id)
#             mean_d = df_c.list[list_index][item][query_row-7:query_row+7].mean()
#             #mean_d = mean_  #插入近期均值 mean = subdf[item][index_-6:index_+6].mean()
#             imputed_d = mean_d
#             if np.isnan(imputed_d):
#                 imputed_d = globalmean_
#
#         else:
#             list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
#             imputed_d = getweightedNanMean(list_)
#         indexdf.loc[row, 'estiY'] = imputed_d
#         # indexdf.loc[row,'localMean'] = np.nanmean(dataMatrix[:, 0])  #局部平均值
#         indexdf.loc[row,'SES'] = ses_d   # 简单线性平滑
#         indexdf.loc[row,'IDW'] = idw_d   # 逆距离加权
#         indexdf.loc[row,'ubcf'] = ubcf_d # 相关性计算
#         indexdf.loc[row,'ibcf'] = ibcf_d  #
#     str_ = str(k) + 'pearson' + 'window' + str(wind)
#     file = str_ + outfile
#     indexdf.to_csv(file)
#     # 写结果到txt中
#     mae = getMAE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     mre = getMRE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     with open('result/result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#         print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     return mae, mre
# def insert_pearsonBased_multi_viewer_estimate(df,indexdf,item,outfile,neighbordf,neighbor_dis_df,knnVal=5,wind=13):
#     #df已经删除测试位置的df
#     # 遍历index，中station_id
#     # check_list =
#     # print(df.loc[4306,:])
#     #indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['localMean'] = pd.Series(np.zeros(len(indexdf)))  # 真实值
#     indexdf['SES'] = pd.Series(np.zeros(len(indexdf)))  # 估计值
#     indexdf['IDW'] = pd.Series(np.zeros(len(indexdf)))  # 真实值
#     indexdf['ubcf'] = pd.Series(np.zeros(len(indexdf)))  # 估计值
#     indexdf['ibcf'] = pd.Series(np.zeros(len(indexdf)))  # 真实值
#     middle = (int)(wind / 2)  # get_SimpleExponentialSmooth window middle
#     if knnVal == -1:
#         k = len(neighbordf)
#     else:
#         k = knnVal
#     globalmean_ = df[item].mean()
#
#
#     df_c = df_clip(df, 'station_id')  # 数据按照station_id划分
#     # mean = df['PM2.5'].mean()
#     # print(mean)
#     # index_c = df_clip(indexdf, 'station_id')  #index_c也
#     # assert is_array_equal(df_c.id, index_c.id), "station sizes are not equal"
#     # size = len(df_c.list)
#     # realY = []
#     # estiY = []
#     columns = 0
#     for i in range(df_c.list[0].columns.size):  # 获取item对应的列号
#         if df_c.list[0].columns.values[i] == item:
#             columns = i
#             break
#
#     for row in indexdf.index.values:  # 逐行便利indexdf
#         if row % 1000 == 0:
#             print(row)
#         # 获取stationid，row号
#         station_id = indexdf.loc[row, 'station_id']
#         # dis = neighbor_dis_df.loc[station_id, :].values[:k]  # get station_id neighbor distance
#         query_row = indexdf.loc[row, 'row']
#         dataMatrix, neighborlist = get_dataMetrix(df_c, neighbordf, station_id, columns, query_row, w=wind, k=k)
#         dis_all = neighbor_dis_df.loc[station_id, :]  # get station_id neighbor distance
#         dis = dis_all[neighborlist]
#         # print(dataMatrix)
#         # print(matrix)
#         # 1
#         ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  # 获得指数平滑值
#         # print('ses_d', ses_d)
#         # get dis array
#         # 2
#         idw_d, idw_d_con = get_InverseDistanceWeighting(dataMatrix[middle, :], dis)  #
#         # print('idw_d', idw_d)
#         # 3
#         ubcf_d, ubcf_d_con = get_UserBasedCollaborativeFiltering(dataMatrix)
#         # print('ubcf_d', ubcf_d)
#         # 4
#         ibcf_d, ibcf_d_con = get_ItembasedCollaborativefiltering(dataMatrix)
#
#         if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):
#             # print(row, column, 'insert all NAN, we replace this by mean')
#             #获取id号对应的list号
#             #寻找到近期的数据数组
#             list_index = df_c.get_listIndex_by_id(station_id)
#             mean_d = df_c.list[list_index][item][query_row-7:query_row+7].mean()
#             #mean_d = mean_  #插入近期均值 mean = subdf[item][index_-6:index_+6].mean()
#             imputed_d = mean_d
#             if np.isnan(imputed_d):
#                 imputed_d = globalmean_
#
#         else:
#             list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
#             imputed_d = getweightedNanMean(list_)
#         indexdf.loc[row, 'estiY'] = imputed_d
#         # indexdf.loc[row,'localMean'] = np.nanmean(dataMatrix[:, 0])  #局部平均值
#         indexdf.loc[row,'SES'] = ses_d   # 简单线性平滑
#         indexdf.loc[row,'IDW'] = idw_d   # 逆距离加权
#         indexdf.loc[row,'ubcf'] = ubcf_d # 相关性计算
#         indexdf.loc[row,'ibcf'] = ibcf_d  #
#     str_ = str(k) + 'pearson' + 'window' + str(wind)
#     file = str_ + outfile
#     indexdf.to_csv(file)
#     # 写结果到txt中
#     mae = getMAE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     mre = getMRE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     with open('result/result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#         print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     return mae, mre
# def my_multi_viewer_estimate(df,indexdf,item,outfile,neighbordf,neighbor_dis_df,knnVal=5,wind=13):
#     #df已经删除测试位置的df
#     # 遍历index，中station_id
#     # check_list =
#     # print(df.loc[4306,:])
#     #indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['localMean'] = pd.Series(np.zeros(len(indexdf)))  #记录local mean
#     indexdf['SES'] = pd.Series(np.zeros(len(indexdf)))  # 记录SES
#     indexdf['IDW'] = pd.Series(np.zeros(len(indexdf)))  # 记录IDW
#     indexdf['ubcf'] = pd.Series(np.zeros(len(indexdf)))  # 记录ubcf
#     indexdf['ibcf'] = pd.Series(np.zeros(len(indexdf)))  # 记录了ibcf
#
#     middle = (int)(wind / 2)  # get_SimpleExponentialSmooth window middle
#     if knnVal == -1:
#         k = len(neighbordf)
#     else:
#         k = knnVal
#     globalmean_ = df[item].mean() #当数据插值失效，采用前一天的数据覆盖
#
#     df_c = df_clip(df, 'station_id')  # 数据按照station_id划分
#     columns = 0  #确认item对应的属性列
#     for i in range(df_c.list[0].columns.size):  # 获取item对应的列号
#         if df_c.list[0].columns.values[i] == item:
#             columns = i
#             break
#     situation_count = [0, 0, 0, 0, 0, 0]
#
#
#     for row in indexdf.index.values:  # 逐行遍历indexdf
#         if row % 1000 == 0:
#             print(row)
#         # 获取stationid，row号
#         station_id = indexdf.loc[row, 'station_id']  # id号
#         # dis = neighbor_dis_df.loc[station_id, :].values[:k]  # get station_id neighbor distance
#         query_row = indexdf.loc[row, 'row'] #行号
#         dataMatrix, neighborlist = get_dataMetrix(df_c, neighbordf, station_id, columns, query_row, w=wind, k=k)  #获取邻域矩阵，和距离index
#         dis_all = neighbor_dis_df.loc[station_id, :]  # get station_id neighbor distance
#         dis = dis_all[neighborlist] #获取Pearson相关系数K近邻的距离列表
#         # print(dataMatrix)
#         # print(matrix)
#         # 1
#         #TODO 插入local mean
#         list_index = df_c.get_listIndex_by_id(station_id)
#         local_mean = df_c.list[list_index][item][query_row - 12:query_row + 12].mean()
#         indexdf.loc[row, 'localMean'] = local_mean
#         #TODO step1 判断数组缺失值数量
#         # series = pd.DataFrame(dataMatrix[:,0])
#         # nan_num = series.isna().sum().values[0]  # 判断第一列nan值的数量
#         # rate = nan_num / len(series)  # nan值的比例
#
#         rate = get_nan_rate(dataMatrix[:,0])  # 获取第一列的nan值的比例
#         # print(rate)
#         #TODO 0824 add大比例数据缺失，使用前一天数据覆盖。
#         # if rate > 4/5:
#         #     pass        #使用一天之前的数据填充
#         if rate < 1/5: #缺失数据比例
#             ses_d,ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.95) #使用ses
#             indexdf.loc[row, 'estiY'] = ses_d
#             #将数据填充回，缺失dataframe #缺失值数量缺失DataFrame中
#             #获得行号，获得列号,回填估计数值
#             list_index = df_c.get_listIndex_by_id(station_id)
#             df_c.list[list_index].loc[query_row,item] = ses_d
#             situation_count[0]+=1  #插入SES
#             indexdf.loc[row, 'SES'] = 1
#         else:
#             #使用全部方法
#             # 1
#             ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  # 获得指数平滑值
#             # print('ses_d', ses_d)
#             # get dis array
#             # 2
#             idw_d, idw_d_con = get_InverseDistanceWeighting(dataMatrix[middle, :], dis)  #
#             # print('idw_d', idw_d)
#             # 3
#             ubcf_d, ubcf_d_con = get_UserBasedCollaborativeFiltering(dataMatrix)
#             # print('ubcf_d', ubcf_d)
#             # 4
#             ibcf_d, ibcf_d_con = get_ItembasedCollaborativefiltering(dataMatrix)
#             # print('ibcf_d', ibcf_d)
#             if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):
#                 # print(row, column, 'insert all NAN, we replace this by mean')
#                 #获取id号对应的list号
#                 #寻找到近期的数据数组
#                 list_index = df_c.get_listIndex_by_id(station_id)
#                 mean_d = df_c.list[list_index][item][query_row-12:query_row+12].mean()
#                 #mean_d = mean_  #插入近期均值 mean = subdf[item][index_-6:index_+6].mean()
#                 imputed_d = mean_d
#
#                 if np.isnan(imputed_d):
#                     imputed_d = globalmean_
#                     situation_count[2] += 1 #插入全局均值
#                 else:
#                     situation_count[1] += 1 #插入近期的均值
#
#
#             else:
#                 list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
#                 imputed_d = getweightedNanMean(list_)
#                 situation_count[3] += 1  # 插入加权均值
#
#
#             indexdf.loc[row, 'estiY'] = imputed_d
#             if not np.isnan(ses_d):
#                 indexdf.loc[row, 'SES'] = 1
#             if not np.isnan(idw_d):
#                 indexdf.loc[row, 'IDW'] = 1
#             if not np.isnan(ubcf_d):
#                 indexdf.loc[row, 'ubcf'] = 1
#             if not np.isnan(ibcf_d):
#                 indexdf.loc[row, 'ibcf'] = 1
#             # 将数据填充回，缺失dataframe #缺失值数量缺失DataFrame中
#             # 获得行号，获得列号,回填估计数值
#             list_index = df_c.get_listIndex_by_id(station_id)
#             df_c.list[list_index].loc[query_row, item] = imputed_d
#
#         # indexdf.loc[row, 'estiY'] = imputed_d
#         # # indexdf.loc[row,'localMean'] = np.nanmean(dataMatrix[:, 0])  #局部平均值
#         # indexdf.loc[row,'SES'] = ses_d   # 简单线性平滑
#         # indexdf.loc[row,'IDW'] = idw_d   # 逆距离加权
#         # indexdf.loc[row,'ubcf'] = ubcf_d # 相关性计算
#         # indexdf.loc[row,'ibcf'] = ibcf_d  #
#     str_ = str(k) + 'pearson' + 'window' + str(wind) + '0907'
#     file = str_ + outfile
#     indexdf.to_csv(file)
#     # 写结果到txt中
#     mae = getMAE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     mre = getMRE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     with open('result/result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#         print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#         print('situation:0:', situation_count[0], '1:', situation_count[1], '2:', situation_count[2],
#           '3:', situation_count[3], '4:', situation_count[4], '5:', situation_count[5], file=fileHandle)
#     return mae, mre
# def my_multi_viewer_estimate_withconfidence(df,indexdf,item,outfile,neighbordf,neighbor_dis_df,knnVal=5,wind=13):
#     # TODO 对于多种插值方法，根据其估计时，缺失值数量，定义置信值
#     #df已经删除测试位置的df
#     # 遍历index，中station_id
#     # check_list =
#     # print(df.loc[4306,:])
#     #indexdf['realY'] = pd.Series(np.zeros(len(indexdf)))
#     # 对indexdf按照row进行升序排序，
#     indexdf = indexdf.sort_values(by='row',ascending=True)
#     indexdf = indexdf.reset_index(drop=True)
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['localMean'] = pd.Series(np.zeros(len(indexdf)))  #记录local mean
#     indexdf['SES'] = pd.Series(np.zeros(len(indexdf)))  # 记录SES
#     indexdf['IDW'] = pd.Series(np.zeros(len(indexdf)))  # 记录IDW
#     indexdf['ubcf'] = pd.Series(np.zeros(len(indexdf)))  # 记录ubcf
#     indexdf['ibcf'] = pd.Series(np.zeros(len(indexdf)))  # 记录了ibcf
#     middle = (int)(wind / 2)  # get_SimpleExponentialSmooth window middle
#     if knnVal == -1:
#         k = len(neighbordf)
#     else:
#         k = knnVal
#     globalmean_ = df[item].mean() #当数据插值失效，采用前一天的数据覆盖
#
#     df_c = df_clip(df, 'station_id')  # 数据按照station_id划分
#     columns = 0  #确认item对应的属性列
#     for i in range(df_c.list[0].columns.size):  # 获取item对应的列号
#         if df_c.list[0].columns.values[i] == item:
#             columns = i
#             break
#     situation_count = [0,0,0,0,0,0]
#     for row in indexdf.index.values:  # 逐行遍历indexdf
#         if row % 1000 == 0:
#             print(row)
#         # 获取stationid，row号
#         station_id = indexdf.loc[row, 'station_id']  # id号
#         list_index = df_c.get_listIndex_by_id(station_id)  #TODO 获取id对应的list序号，可查找全部数据
#         # dis = neighbor_dis_df.loc[station_id, :].values[:k]  # get station_id neighbor distance
#         query_row = indexdf.loc[row, 'row'] #行号
#         dataMatrix, neighborlist = get_dataMetrix(df_c, neighbordf, station_id, columns, query_row, w=wind, k=k)  #获取邻域矩阵，和距离index
#         dis_all = neighbor_dis_df.loc[station_id, :]  # get station_id neighbor distance
#         dis = dis_all[neighborlist] #获取Pearson相关系数K近邻的距离列表
#         # print(dataMatrix)
#         # print(matrix)
#         # 1
#         #TODO step1 判断数组缺失值数量
#         # series = pd.DataFrame(dataMatrix[:,0])
#         # nan_num = series.isna().sum().values[0]  # 判断第一列nan值的数量
#         column_datalossrate = get_nan_rate(dataMatrix[:,0])  # 获取该列的nan值的比例
#         row_datalossrate = get_nan_rate(dataMatrix[middle,:]) # 获取改行的nan值比例
#         # series = pd.DataFrame(dataMatrix[middle,:])
#         imputed_d = 0
#         succeed = False
#         if row_datalossrate == 1: #TODO 行全为nan,多视图插值 只有SES可以使用
#             # if column_datalossrate > 3/4: #TODO SES也失效
#             #     indexdf.loc[row, 'estiY'] = globalmean_  #全局均值
#             #     df_c.list[list_index].loc[query_row, item] = globalmean_  # 插入总的数据集中 #ses #插入历史均值
#             # else:  #
#             #     ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  # 使用ses
#             #     imputed_d = getweightNanMeanWithConfidence([ses_d,globalmean_],[])
#             #     indexdf.loc[row, 'estiY'] = ses_d
#             #     df_c.list[list_index].loc[query_row, item] = ses_d  # 插入总的数据集中 #ses
#             #TODO 行数据全为空时,多视图插值方法,除ses外其他失效,所以最终估计为ses与全局均值的置信值加权平均,如果ses也为nan,最终估计为global mean
#             ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  # 使用ses
#             if np.isnan(ses_d):  #ses值失效
#                 #使用一天前的数值替代
#                 last_row = query_row
#                 while last_row - 24 >=0: #往前日寻找   #数据缺失严重
#                     last_row -= 24
#                     if not np.isnan(df_c.list[list_index].loc[last_row, item]):
#                         imputed_d = df_c.list[list_index].loc[last_row, item]
#                         succeed = True
#                         break
#                         situation_count[0] +=1 #TODO situation_count[0] row全空,ses估计失败,循环前一天时刻存在存在
#                 if not succeed:
#                     imputed_d = globalmean_
#                     succeed = True
#                     situation_count[1] += 1 #TODO situation_count[1] row全空,ses估计失败,循环前一天数据都为nan
#
#                 indexdf.loc[row, 'estiY'] = imputed_d
#                 indexdf.loc[row, 'localMean'] = imputed_d  # 记录local mean or global mean
#                 df_c.list[list_index].loc[query_row, item] = imputed_d  # 插入总的数据集中 #ses
#
#             else:  #ses值有效
#                 imputed_d = ses_d #getweightNanMeanWithConfidence([ses_d,globalmean_],[ses_d_con,1])  #TODO *创新点*
#                 succeed = True
#                 situation_count[2] += 1  # TODO situation_count[2] row全空,ses估计成功
#                 indexdf.loc[row, 'estiY'] = imputed_d
#                 indexdf.loc[row, 'SES'] = ses_d  # 记录SES
#                 indexdf.loc[row, 'localMean'] = globalmean_  # 记录local mean
#                 df_c.list[list_index].loc[query_row, item] = imputed_d  # 插入总的数据集中 #ses
#
#
#         else:  #TODO 行不全为nan,多视图插值可能可以使用
#             #正常操作
#             # if column_datalossrate < 1/4: #TODO 列的缺失值较少，则可以使用ses估计，取得最优解
#             #     ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  # 使用ses
#             #     indexdf.loc[row, 'estiY'] = ses_d
#             #     df_c.list[list_index].loc[query_row, item] = ses_d  # 插入总的数据集中
#             #
#             #     indexdf.loc[row,'SES'] = ses_d # 记录SES
#             #
#             # else: #TODO 列缺失值较多 直接加载全部方法
#             ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  # 获得指数平滑值
#             # print('ses_d', ses_d)
#             # get dis array
#             # 2
#             idw_d, idw_d_con = get_InverseDistanceWeighting(dataMatrix[middle, :], dis)  #
#             # print('idw_d', idw_d)
#             # 3
#             ubcf_d, ubcf_d_con = get_UserBasedCollaborativeFiltering(dataMatrix)
#             # print('ubcf_d', ubcf_d)
#             # 4
#             ibcf_d, ibcf_d_con = get_ItembasedCollaborativefiltering(dataMatrix)
#             # print('ibcf_d', ibcf_d)
#
#
#             if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):  # TODO 全为空, 数据缺失导致多视图插值失效
#                 # print(row, column, 'insert all NAN, we replace this by mean')
#                 # 寻找到近期的数据数组
#                 begin = 0
#                 end = 0
#                 if query_row - 12 < 0:  # 判断是否越界  #
#                     begin = 0
#                 if query_row + 12 > len(df_c.list[list_index]):
#                     end = len(df_c.list[list_index])
#                 imputed_d = df_c.list[list_index][item][begin:end].mean()
#                 # mean_d = mean_  #插入近期均值 mean = subdf[item][index_-6:index_+6].mean()
#                 #imputed_d = mean_d
#                 if np.isnan(imputed_d):  # TODO 插入值为nan
#                     imputed_d = globalmean_
#                     situation_count[4] += 1  # TODO situation_count[4] row不全空,近期一天数据全空,插入全局均值
#                 else:
#                     situation_count[3] += 1  # TODO situation_count[3] row不全空,近期一天数据不全空,插入均值
#                 indexdf.loc[row, 'estiY'] = imputed_d
#                 indexdf.loc[row, 'localMean'] = mean_d  # 记录local mean
#                 df_c.list[list_index].loc[query_row, item] = imputed_d  # 插入总的数据集中 #ses
#
#             else:  # TODO 多视图插值方法不全为nan
#                 list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
#                 # using 置信值加权平均
#                 confidence = [ses_d_con, idw_d_con, ubcf_d_con, ibcf_d_con]
#                 imputed_d = getweightNanMeanWithConfidence(list_, confidence)
#
#                 # imputed_d = getweightedNanMean2(list_)
#
#                 indexdf.loc[row, 'estiY'] = imputed_d
#                 indexdf.loc[row, 'SES'] = ses_d  # 记录SES
#                 indexdf.loc[row, 'IDW'] = idw_d  # 记录IDW
#                 indexdf.loc[row, 'ubcf'] = ubcf_d  # 记录ubcf
#                 indexdf.loc[row, 'ibcf'] = ibcf_d  # 记录了ibcf
#                 # 将数据填充回，缺失dataframe #缺失值数量缺失DataFrame中
#                 # 获得行号，获得列号,回填估计数值
#                 # list_index = df_c.get_listIndex_by_id(station_id)
#                 df_c.list[list_index].loc[query_row, item] = imputed_d
#                 situation_count[5] += 1  # TODO situation_count[3] row不全空,近期一天数据不全空,插入均值
#         # print(rate)
#         #TODO 0824 add大比例数据缺失，使用前一天数据覆盖。
#         # if rate > 4/5:
#         #     pass        #使用一天之前的数据填充
#
#         # 判断该列数据缺失量
#         # if column_datalossrate > 1 / 2:  # 一半多的数据为空，表示存在该时许这段时间发生全部缺失。行缺失
#         #     temp_row = query_row
#         #     impute_d = np.nan
#         #     while (temp_row - 24 > 0 and df_c.list[list_index].loc[temp_row - 24, 'station_id'] == station_id):
#         #         if not np.isnan(df_c.list[list_index].loc[temp_row - 24, item]):
#         #             impute_d = df_c.list[list_index].loc[temp_row - 24, item]
#         #         else:
#         #             temp_row -= 24
#         #     if np.isnan(impute_d):  # 部分开始数据，往前推24个时间点，可能会出区域
#         #         impute_d = globalmean_
#         #
#         #     indexdf.loc[row, 'estiY'] = impute_d
#         #     df_c.list[list_index].loc[query_row, item] = impute_d  # 插入总的数据集中
#         # if column_datalossrate < 1/4: #缺失数据比例
#         #     ses_d,ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8) #使用ses
#         #     indexdf.loc[row, 'estiY'] = ses_d
#         #     #将数据填充回，缺失dataframe #缺失值数量缺失DataFrame中
#         #     #获得行号，获得列号,回填估计数值
#         #     #list_index = df_c.get_listIndex_by_id(station_id)#放在函数开始
#         #     df_c.list[list_index].loc[query_row,item] = ses_d  #插入总的数据集中
#         # else:
#         #     #使用全部方法
#         #     #if 估计值还是nan, 使用local mean
#         #
#         #     # 1
#         #     ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=0.8)  # 获得指数平滑值
#         #     # print('ses_d', ses_d)
#         #     # get dis array
#         #     # 2
#         #     idw_d, idw_d_con = get_InverseDistanceWeighting(dataMatrix[middle, :], dis)  #
#         #     # print('idw_d', idw_d)
#         #     # 3
#         #     ubcf_d, ubcf_d_con = get_UserBasedCollaborativeFiltering(dataMatrix)
#         #     # print('ubcf_d', ubcf_d)
#         #     # 4
#         #     ibcf_d, ibcf_d_con = get_ItembasedCollaborativefiltering(dataMatrix)
#         #     # print('ibcf_d', ibcf_d)
#         #     if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):
#         #         # print(row, column, 'insert all NAN, we replace this by mean')
#         #         #获取id号对应的list号
#         #         #寻找到近期的数据数组
#         #         list_index = df_c.get_listIndex_by_id(station_id)
#         #         mean_d = df_c.list[list_index][item][query_row-7:query_row+7].mean()
#         #         #mean_d = mean_  #插入近期均值 mean = subdf[item][index_-6:index_+6].mean()
#         #         imputed_d = mean_d
#         #         if np.isnan(imputed_d):
#         #             imputed_d = globalmean_
#         #
#         #     else:
#         #         list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
#         #         confidence = [ses_d_con, idw_d_con, ubcf_d_con, ibcf_d_con]
#         #         imputed_d = getweightNanMeanWithConfidence(list_,confidence)
#         #
#         #     indexdf.loc[row, 'estiY'] = imputed_d
#         #     # 将数据填充回，缺失dataframe #缺失值数量缺失DataFrame中
#         #     # 获得行号，获得列号,回填估计数值
#         #     # list_index = df_c.get_listIndex_by_id(station_id)
#         #     df_c.list[list_index].loc[query_row, item] = imputed_d
#
#         # indexdf.loc[row, 'estiY'] = imputed_d
#         # # indexdf.loc[row,'localMean'] = np.nanmean(dataMatrix[:, 0])  #局部平均值
#         # indexdf.loc[row,'SES'] = ses_d   # 简单线性平滑
#         # indexdf.loc[row,'IDW'] = idw_d   # 逆距离加权
#         # indexdf.loc[row,'ubcf'] = ubcf_d # 相关性计算
#         # indexdf.loc[row,'ibcf'] = ibcf_d  #
#     str_ = str(k) + 'pearson' + 'window' + str(wind)
#     file = str_ + outfile
#     indexdf.to_csv(file)
#     # 写结果到txt中
#     mae = getMAE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     mre = getMRE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     with open('result/result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#         print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#         print('situation:0:',situation_count[0],'1:',situation_count[1],'2:',situation_count[2],
#               '3:',situation_count[3],'4:',situation_count[4],'5:',situation_count[5], file=fileHandle)
#     return mae, mre

# def my_estimate(df,indexdf,item,outfile,neighbordf,neighbor_dis_df,knnVal=5,wind=13):
#     #TODO
#     # df已经删除测试位置的df
#     # 遍历index，中station_id
#     indexdf['estiY'] = pd.Series(np.zeros(len(indexdf)))
#     indexdf['localMean'] = pd.Series(np.zeros(len(indexdf)))  # 记录local mean
#     indexdf['SES'] = pd.Series(np.zeros(len(indexdf)))  # 记录SES
#     indexdf['IDW'] = pd.Series(np.zeros(len(indexdf)))  # 记录IDW
#     indexdf['ubcf'] = pd.Series(np.zeros(len(indexdf)))  # 记录ubcf
#     indexdf['ibcf'] = pd.Series(np.zeros(len(indexdf)))  # 记录了ibcf
#     bata_ = 0.8  #for SES
#     mean_confidence = bata_*(1-bata_)
#     middle = (int)(wind / 2)  # get_SimpleExponentialSmooth window middle
#     if knnVal == -1:
#         k = len(neighbordf)
#     else:
#         k = knnVal
#     globalmean_ = df[item].mean()  # 当数据插值失效，采用前一天的数据覆盖
#
#     df_c = df_clip(df, 'station_id')  # 数据按照station_id划分
#     columns = 0  # 确认item对应的属性列
#     for i in range(df_c.list[0].columns.size):  # 获取item对应的列号
#         if df_c.list[0].columns.values[i] == item:
#             columns = i
#             break
#     situation_count = [0, 0, 0, 0, 0, 0]
#     for row in indexdf.index.values:  # 逐行遍历indexdf
#         if row % 1000 == 0:
#             print(row)
#         # 获取stationid，row号
#         station_id = indexdf.loc[row, 'station_id']  # id号
#         # dis = neighbor_dis_df.loc[station_id, :].values[:k]  # get station_id neighbor distance
#         query_row = indexdf.loc[row, 'row'] #行号
#         dataMatrix, neighborlist = get_dataMetrix(df_c, neighbordf, station_id, columns, query_row, w=wind, k=k)  #获取邻域矩阵，和距离index
#         dis_all = neighbor_dis_df.loc[station_id, :]  # get station_id neighbor distance
#         dis = dis_all[neighborlist] #获取Pearson相关系数K近邻的距离列表
#         # print(dataMatrix)
#         # print(matrix)
#         # 1
#         #TODO 插入local mean
#         list_index = df_c.get_listIndex_by_id(station_id)
#         local_mean = df_c.list[list_index][item][query_row - 12:query_row + 12].mean()
#         if np.isnan(local_mean):
#             local_mean = globalmean_
#             situation_count[5] += 1  # 使用全局均值
#         else:
#             situation_count[4] += 1  # 使用近期的均值
#
#         indexdf.loc[row, 'localMean'] = local_mean
#         # TODO step1 判断数组缺失值数量
#         column_nanrate = get_nan_rate(dataMatrix[:, 0])  # 获取第一列的nan值的比例
#         row_nanrate = get_nan_rate(dataMatrix[middle, :])  # 获取待估计行的nan值的比例
#         imputed_d = np.NaN
#         if column_nanrate < 1: #该节点，当前时间窗口内的数据没有全缺失
#             if row_nanrate < 1: #待估计节点时间窗口内数据没有全部缺失;待估计时间点，全行数据没有全缺失
#
#                 #TODO multi insert
#                 # 1
#                 ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=bata_)  # 获得指数平滑值
#                 # 2
#                 idw_d, idw_d_con = get_InverseDistanceWeighting(dataMatrix[middle, :], dis)  #
#                 # 3
#                 ubcf_d, ubcf_d_con = get_UserBasedCollaborativeFiltering(dataMatrix)
#                 # 4
#                 ibcf_d, ibcf_d_con = get_ItembasedCollaborativefiltering(dataMatrix)
#                 if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):
#                     #list_index = df_c.get_listIndex_by_id(station_id)
#                     #mean_d = df_c.list[list_index][item][query_row - 12:query_row + 12].mean()
#                     # mean_d = mean_  #插入近期均值 mean = subdf[item][index_-6:index_+6].mean()
#                     imputed_d = local_mean #上文求解 无法进入
#                 else: #多视图插值有效
#                     list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
#                     con = [ses_d_con,idw_d_con,ubcf_d_con,ibcf_d_con]
#                     # imputed_d = getweightedNanMean(list_) #加权平均
#                     imputed_d = getweightNanMeanWithConfidence(list_,con) #加权平均
#
#                 #if not np.isnan(ses_d):
#                 indexdf.loc[row, 'SES'] = ses_d
#                 #if not np.isnan(idw_d):
#                 indexdf.loc[row, 'IDW'] = idw_d
#                 #if not np.isnan(ubcf_d):
#                 indexdf.loc[row, 'ubcf'] = ubcf_d
#                 #if not np.isnan(ibcf_d):
#                 indexdf.loc[row, 'ibcf'] = ibcf_d
#                 situation_count[0] += 1  # 6000
#             else:   #待估计节点时间窗口内数据没有全部缺失;待估计时间点全行数据全缺失
#
#                 situation_count[1] += 1  #7000
#                 ses_d, ses_d_con = get_SimpleExponentialSmooth(dataMatrix[:, 0], bata=bata_)  # 获得指数平滑值
#                 # imputed_d = ses_d
#                 list_ = [ses_d,local_mean]
#                 con = [ses_d_con, mean_confidence]
#                 imputed_d = getweightNanMeanWithConfidence(list_,con)#(ses_d * ses_d_con + local_mean * mean_confidence)/(ses_d_con + mean_confidence)
#                 #加权平均，由于PM2.5 每天的变化过大，所以采用近期ses与localmean加权平均的方式
#                 indexdf.loc[row, 'SES'] = ses_d
#                 # TODO 考虑均值情况
#                 #list_index = df_c.get_listIndex_by_id(station_id)
#                 #mean_d = df_c.list[list_index][item][query_row - 12:query_row + 12].mean()
#                 #imputed_d = ses_d * mean_d /2
#         else:  #该节点，当前时间窗口内的数据全缺失 column_nanrate = 1
#             if row_nanrate < 1:  # #待估计节点时间窗口内数据全部缺失;待估计时间点，全行数据没有全缺失
#                 situation_count[2] += 1   #5000
#                 idw_d, idw_d_con = get_InverseDistanceWeighting(dataMatrix[middle, :], dis)  #
#                 imputed_d = idw_d
#                 indexdf.loc[row, 'IDW'] = idw_d
#             else:  #待估计节点时间窗口内数据全部缺失;待估计时间点，全行数据全缺失
#                 a = 1
#                 #TODO 采用前几个的同一时刻的数值均值替代，t t-24 t-24*2 ... t-24*n 不佳
#                 # 判断t-24*n是否越界
#                 # situation_count[3] += 1  # 14000
#                 # his_list = []
#                 # temp_row = query_row
#                 # samplesize = 7
#                 # while temp_row-24 >= 0 and samplesize > 0:  # 查找前7日相同时刻数值，求均值插入
#                 #     his_list.append(df_c.list[list_index].loc[temp_row-24,item])
#                 #     temp_row -= 24
#                 #     samplesize -= 1
#                 # if len(his_list) == 0:
#                 #     #索引位置靠前，无插入值
#                 #     imputed_d = local_mean
#                 # else:
#                 #     rate = get_nan_rate(his_list)
#                 #     if rate == 1:
#                 #         imputed_d = local_mean
#                 #     else:
#                 #         his_mean = np.nanmean(his_list)  # 可能全部插入nan 值
#                 #         imputed_d = his_mean
#
#                 #TODO 2使用前一日的数据
#                 # situation_count[3] += 1  # 14000
#                 # #his_list = []
#                 # temp_row = query_row
#                 # insert_label = False
#                 # while temp_row-24 >= 0 :  # 查找前7日相同时刻数值，求均值插入
#                 #     if np.isnan(df_c.list[list_index].loc[temp_row-24,item]):
#                 #         temp_row -= 24
#                 #     else:
#                 #         imputed_d = df_c.list[list_index].loc[temp_row-24,item]
#                 #         insert_label = True
#                 #         break
#                 # if not insert_label: #
#                 #     imputed_d = local_mean
#
#                 #TODO 3 使用
#                 pass
#
#
#
#         indexdf.loc[row, 'estiY'] = imputed_d
#         # if np.isnan(imputed_d):
#         #     print('test')
#     str_ = 'my'+'shape_'+str(k)+'_'+str(wind)
#     file = str_ + outfile
#     # mae = getMAE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     # mre = getMRE(indexdf['data'].tolist(), indexdf['estiY'].tolist())
#     indexdf.to_csv(file)
#
#
#     with open('result/result1.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#         #print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#         print('situation:0:', situation_count[0], '1:', situation_count[1], '2:', situation_count[2],
#               '3:', situation_count[3], '4:', situation_count[4], '5:', situation_count[5], file=fileHandle)
#     #return mae, mre
#     return 0,0

# def run_17_18():
    ##TODO 过滤离群值

    ##TODO  获取测试数据位置 op1 # run once, output:origin/checkloss17_18_aq.csv
    # run once
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # check_loss(data, 'PM2.5', 'origin/checkloss17_18_aq.csv')

    ##TODO 获取删除测试位置的数据 op2 run once, output:origin/del_testdata_17_18_PM2_5.csv
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # del_data_by_index(data, check_list, 'PM2.5', '17_18_PM2_5.csv')


    # 计算不同插值方法的mre mae值 op3
    # 清空记录文件
    # with open('result.txt', mode='w') as fileHandle:  # 'w'写，'r'读，'a'添加
    #     pass #clear txt

    ###TODO  globalmean 方法 op3.1
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # #oped_data = pd.read_csv("../111CompareImputer/del_testdata_17_18_PM2_5.csv")
    # mae, mre = insert_global_mean(data,check_list,'PM2.5','17_18_PM2_5.csv')
    # print(mae,mre)

    ##TODO localmean 方法 op3.2
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # w = 13
    # mae, mre = insert_history_mean(data,oped_data,check_list,'PM2.5',w,'17_18_PM2_5.csv')
    # print(mae, mre)

    ##TODO xgboost方法 op3.3 xgboost
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # # data = data[(data['station_id'] =='aotizhongxin_aq') | (data['station_id'] =='badaling_aq')]
    #
    # ##mae, mre = insert_xgboost(data, check_list, 'PM2.5', '17_18_PM2_5.csv') ##替换为下一行
    # mae, mre = insert_data_bymethod(data, oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv','xgboost')
    # print(mae, mre)

    ##TODO lightbgm方法 op3.4
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # # data = data[(data['station_id'] =='aotizhongxin_aq') | (data['station_id'] =='badaling_aq')]
    #
    # # mae, mre = insert_lightbgm(data, check_list, 'PM2.5', '17_18_PM2_5.csv')#替换为下一行
    # mae, mre = insert_data_bymethod(data,oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv','lightgbm')
    # print(mae, mre)

    ##TODO knn op3,5
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list[' station_id'] =='badaling_aq')]
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # # data = data[(data['station_id'] =='aotizhongxin_aq') | (data['station_id'] =='badaling_aq')]
    # #mae, mre = insert_knn(data, check_list, 'PM2.5', '17_18_PM2_5.csv') #替换为下一行
    # mae, mre = insert_data_bymethod(data,oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv','knn')
    # print(mae, mre)

    ##TODO randomforest op3,6 # 太慢
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # mae, mre = insert_data_bymethod(data, oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv','randomforest')
    # print(mae, mre)

    # TODO multiviewer方法 op3.7 转至st_viewer.py
    #
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
    # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv',neighbor_info, neighbor_dis_info,knnVal=-1,wind=7)
    # print(mae, mre)
    #
    ## #TODO##multiviewer knearest方法 op3.8

    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # #data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
    # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # mae, mre = insert_multi_viewer_estimate( oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
    #                                        neighbor_dis_info, knnVal=10, wind=7)
    # print(mae, mre)

    # ##TODO multiviewer knearest方法 op3.9
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
    # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
    #                                         neighbor_dis_info, knnVal=5, wind=7)
    # print(mae, mre)

    ##TODO multiviewer knearest方法 op3.9 wind=11
    #
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # #data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
    # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
    #                                         neighbor_dis_info, knnVal=5, wind=11)
    # print(mae, mre)
    ##TODO bata 改为0.8 op3.10 wind=15
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
    # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
    #                                         neighbor_dis_info, knnVal=5, wind=15)
    # print(mae, mre)

    ##TODO pearson based multiviewer based method op3.11
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # #data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data,dis_df,'PM2.5','17_18')
    # # pearson_neighbor_info = 0
    # # pearson_neighbor_dis = 0
    # # neighbor_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
    # # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # mae, mre = my_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
    #                                         neighbor_dis_info, knnVal=5+1, wind=15) #第一列为自身
    # print(mae, mre)

    ##TODO pearson based multiviewer based method op3.12
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data, dis_df, 'PM2.5', '17_18')
    # # pearson_neighbor_info = 0
    # # pearson_neighbor_dis = 0
    # # neighbor_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
    # # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # mae, mre = my_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
    #                                     neighbor_dis_info, knnVal=5 + 1, wind=9)  # 第一列为自身
    # print(mae, mre)
    #

    ##TODO pearson based multiviewer based method op3.13 回填机制 效果不佳
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data, dis_df, 'PM2.5', '17_18')
    # # pearson_neighbor_info = 0
    # # pearson_neighbor_dis = 0
    # # neighbor_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
    # # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # mae, mre = my_multi_viewer_estimate_withconfidence(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
    #                                     neighbor_dis_info, knnVal=5 + 1, wind=11)  # 第一列为自身
    # print(mae, mre)

    # TODO 0908 my_estimate

    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data, dis_df, 'PM2.5', '17_18')
    # # pearson_neighbor_info = 0
    # # pearson_neighbor_dis = 0
    # # neighbor_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
    # # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # mae, mre = my_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info, neighbor_dis_info, knnVal=10 , wind=13)  # 第一列为自身
    # print(mae, mre)
    # pass

#TODO 待修改
# def run_14_15():
#     ##TODO 过滤离群值
#
#     ##TODO  获取测试数据位置 op1 # run once, output:origin/checkloss17_18_aq.csv
#     # run once
#     data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     check_loss(data, 'PM2.5', 'origin/checkloss17_18_aq.csv')
#
#     ##TODO 获取删除测试位置的数据 op2 run once, output:origin/del_testdata_17_18_PM2_5.csv
#     check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     del_data_by_index(data, check_list, 'PM2.5', '17_18_PM2_5.csv')
#     # 计算不同插值方法的mre mae值 op3
#     # 清空记录文件
#     # with open('result.txt', mode='w') as fileHandle:  # 'w'写，'r'读，'a'添加
#     #     pass #clear txt
#
#     ###TODO  globalmean 方法 op3.1
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # #oped_data = pd.read_csv("../111CompareImputer/del_testdata_17_18_PM2_5.csv")
#     # mae, mre = insert_global_mean(data,check_list,'PM2.5','17_18_PM2_5.csv')
#     # print(mae,mre)
#
#     ##TODO localmean 方法 op3.2
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # w = 13
#     # mae, mre = insert_history_mean(data,oped_data,check_list,'PM2.5',w,'17_18_PM2_5.csv')
#     # print(mae, mre)
#
#     ##TODO xgboost方法 op3.3 xgboost
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
#     # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # # data = data[(data['station_id'] =='aotizhongxin_aq') | (data['station_id'] =='badaling_aq')]
#     #
#     # ##mae, mre = insert_xgboost(data, check_list, 'PM2.5', '17_18_PM2_5.csv') ##替换为下一行
#     # mae, mre = insert_data_bymethod(data, oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv','xgboost')
#     # print(mae, mre)
#
#     ##TODO lightbgm方法 op3.4
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
#     # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # # data = data[(data['station_id'] =='aotizhongxin_aq') | (data['station_id'] =='badaling_aq')]
#     #
#     # # mae, mre = insert_lightbgm(data, check_list, 'PM2.5', '17_18_PM2_5.csv')#替换为下一行
#     # mae, mre = insert_data_bymethod(data,oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv','lightgbm')
#     # print(mae, mre)
#
#     ##TODO knn op3,5
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
#     # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # # data = data[(data['station_id'] =='aotizhongxin_aq') | (data['station_id'] =='badaling_aq')]
#     # #mae, mre = insert_knn(data, check_list, 'PM2.5', '17_18_PM2_5.csv') #替换为下一行
#     # mae, mre = insert_data_bymethod(data,oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv','knn')
#     # print(mae, mre)
#
#     ##TODO randomforest op3,6 #
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
#     # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # mae, mre = insert_data_bymethod(data, oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv','randomforest')
#     # print(mae, mre)
#
#     # TODO multiviewer方法 op3.7 转至st_viewer.py
#     #
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
#     # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv',neighbor_info, neighbor_dis_info,knnVal=-1,wind=7)
#     # print(mae, mre)
#     #
#     ## #TODO##multiviewer knearest方法 op3.8
#
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # #data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
#     # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # mae, mre = insert_multi_viewer_estimate( oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
#     #                                        neighbor_dis_info, knnVal=10, wind=7)
#     # print(mae, mre)
#
#     # ##TODO multiviewer knearest方法 op3.9
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
#     # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
#     #                                         neighbor_dis_info, knnVal=5, wind=7)
#     # print(mae, mre)
#
#     ##TODO multiviewer knearest方法 op3.9 wind=11
#     #
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # #data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
#     # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
#     #                                         neighbor_dis_info, knnVal=5, wind=11)
#     # print(mae, mre)
#     ##TODO bata 改为0.8 op3.10 wind=15
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
#     # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
#     #                                         neighbor_dis_info, knnVal=5, wind=15)
#     # print(mae, mre)
#
#     ##TODO pearson based multiviewer based method op3.11
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # #data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data,dis_df,'PM2.5','17_18')
#     # # pearson_neighbor_info = 0
#     # # pearson_neighbor_dis = 0
#     # # neighbor_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
#     # # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # mae, mre = my_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
#     #                                         neighbor_dis_info, knnVal=5+1, wind=15) #第一列为自身
#     # print(mae, mre)
#
#     ##TODO pearson based multiviewer based method op3.12
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data, dis_df, 'PM2.5', '17_18')
#     # # pearson_neighbor_info = 0
#     # # pearson_neighbor_dis = 0
#     # # neighbor_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
#     # # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # mae, mre = my_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
#     #                                     neighbor_dis_info, knnVal=5 + 1, wind=9)  # 第一列为自身
#     # print(mae, mre)
#     #
#
#     ##TODO pearson based multiviewer based method op3.13 回填机制 效果不佳
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data, dis_df, 'PM2.5', '17_18')
#     # # pearson_neighbor_info = 0
#     # # pearson_neighbor_dis = 0
#     # # neighbor_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
#     # # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # mae, mre = my_multi_viewer_estimate_withconfidence(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info,
#     #                                     neighbor_dis_info, knnVal=5 + 1, wind=11)  # 第一列为自身
#     # print(mae, mre)
#
#     # TODO 0908 my_estimate
#
#     # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
#     # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
#     # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
#     # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data, dis_df, 'PM2.5', '17_18')
#     # # pearson_neighbor_info = 0
#     # # pearson_neighbor_dis = 0
#     # # neighbor_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_neighbor.csv", index_col='Unnamed: 0')
#     # # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
#     # mae, mre = my_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv', neighbor_info, neighbor_dis_info, knnVal=10 , wind=13)  # 第一列为自身
#     # print(mae, mre)
#     pass
#
if __name__ == "__main__":
    # run_17_18()
    # run_14_15()
    pass

