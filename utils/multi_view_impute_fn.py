#使用sklearn 中的分类器 插入数据
# from ModelBasedImputer.MissingImputer import MissingImputer
# from sklearn.experimental import enable_iterative_imputer #迭代插入方法使能
# from sklearn.impute import IterativeImputer #迭代插入方法
# import pandas as pd
# import numpy as np
# import os
# import joblib
# from Structure import
import pandas as pd
from utils.Structure import df_clip
import numpy as np
from math import sqrt,tanh
import os
np.random.seed(11)

# define multi viewer data imputer function
def get_dataMetrix(df_clip, neighbor_df, impute_name, columns, index, w, k):
    #TODO 20200817,添加排除空列数据操作
    # get data matrix [w x k] for each parameter which is nan. and the nan value is at [w/2,0]
    size = len(df_clip.list[0]) #station_id size
    middle = index  #行号
    #TODO 0817 由于存在数据缺失，所以最近邻可能存在nan值较多情况，通过matrix_list来记录添加到matrix中的序列号,alternative_list记录备选序列号
    matrix_list = []
    alternative_list = []
    # judge the boundary
    if index < w / 2:  #判断下边界
        middle = int(w / 2) #数据的开始
    elif index > size - 1 - int(w / 2): #判断数据的结尾
        middle = size - 1 - int(w / 2)

    begin = int(middle - (w - 1) / 2) #获取时间窗口的两侧
    end = int(middle + (w - 1) / 2)   #
    #     print(begin,' ',middle,' ', end)
    firstLine = True   #输出数据时，判断空操作
    matrix = pd.DataFrame([]) #存储领域矩阵
    alternative_matrix = pd.DataFrame([])  #储存数据量较少的列，当matrix矩阵未满的适合，填充alternative_matrix的前N列
    #TODO  0817
    # neighbor_l = neighbor_df.loc[impute_name, :][0:k]  #获取neighbor_l 时间区域的
    # 先获取全部的领域信息，然后根据领域信息，逐个判断列是否为空，填充列值多余2/3的列
    neighbor_l = neighbor_df.loc[impute_name, :][0 : 2*k] # 获取neighbor_l 时间区域的2k个列的信息
    #     print(neighbor_l)
    # get neighbor index list
    #neighbor_index = []  #返回station——id对应的list index号
    for i, id_ in enumerate(neighbor_l,start=0):   #遍历领域的id名
        # neighbor_index.append(df_clip.get_listIndex_by_id(n))
        #     print(neighbor_index)
        # print(i)
        list_index = df_clip.get_listIndex_by_id(id_)
        series = df_clip.list[list_index].iloc[begin:end + 1, int(columns):int(columns) + 1].copy()
        nan_num = series.isna().sum().values[0] #该列nan值的数量
        nan_rate = nan_num/len(series)  #nan值的比例

        if firstLine:  #第一列插入，且nan的比例小于3/5
            matrix = series
            matrix_list.append(i)  #记录数据记录的index顺序
            #print(matrix)
            firstLine = False
        elif not firstLine and nan_rate < 1:  #其他列插入，且nan值比例小于1,即非空
            matrix = pd.concat([matrix, series], axis=1) #铸列添加
            matrix_list.append(i)   #记录数据记录的index顺序s
        else: #rate= 1 全空情况，添加数据到备选矩阵
            if alternative_matrix.shape[1] == 0:
                alternative_matrix = series
            else:
                alternative_matrix = pd.concat([alternative_matrix, series], axis=1)  # 逐列添加
            alternative_list.append(i)  #按照neighbor_info 填充neighbor信息
        if matrix.shape[1] == k:#提前判断结束情况
            break
    if matrix.shape[1] < k:  # 判断matrix是否已经填满
        required_col_num = k - matrix.shape[1]
        matrix = pd.concat([matrix,alternative_matrix.iloc[:,:required_col_num]],axis=1) #按列拼接
        matrix_list.extend(alternative_list[:required_col_num])  #拼接列表
    return matrix.values, matrix_list

def get_SimpleExponentialSmooth(series, bata):
    # middle
    # V(t) = sum( V(t-i) * bata * (1-bata)^(t-i) ) / sum( bata * (1-bata)^(t-i) ) = numerator/denominator
    # TODO 返回估计值，及置信度(nan值的比例)
    # return
    if np.isnan(series).all(): #数据全为nan,
        #         print('all nan')
        return np.nan,0
    else:
        #         print('not all nan')
        size = len(series) #series的长度
        middle = int(size / 2)  #中间位置
        numerator = 0  #分子
        denominator = 0 #分母
        # nan_count = 0
        con_sum = 0

        for i in range(size):
            temp  = bata * (1 - bata) ** (abs(i - middle)-1)  #当i-middle时，幂值为-1，其他数据与中间间隔个数-1
            if i != middle: #排除待插入位置自身的权重
                con_sum +=  temp #计算总的权重
            if not np.isnan(series[i]):
            #    nan_count += 1
            #else:
                #                 print('temp',temp)
                denominator += temp
                #                 print('denominator',denominator)
                numerator += temp * series[i]

        #                 print('numerator',numerator)
        #         print('Estimator data:',numerator/denominator)
        if denominator == 0: #
            return np.nan, 0
        else:
            # return numerator / denominator, 1 - nan_count/size #返回比例值不合适
            return numerator / denominator, denominator/con_sum  #权重比例用于计算可信度

def get_InverseDistanceWeighting(arr, dis):
    # arr data array
    # dis distance array
    # TODO 返回估计值，及置信度(nan值的比例)
    assert len(dis) == len(arr)
    nan_count = 0
    size = len(dis)
    dis = np.array(dis)
    arr = np.array(arr)
    delete = []
    con_sum = 0
    dis_sum = 0
    weighted_sum = 0 # 加权的数值求和
    for i in range(len(dis)):
        if dis[i] == 0:  # 删除 数组中为nan值，或者距离为0（自身）
            pass
        elif np.isnan(arr[i]): #删除 数组中为nan值，或者距离为0（自身）
            con_sum += dis[i] ** -2
        else:
            dis_sum += dis[i] ** -2
            con_sum += dis[i] ** -2
            weighted_sum += (arr[i] * (dis[i] ** -2))
    if dis_sum == 0:
        return np.nan,0
    else:
         return weighted_sum/dis_sum, dis_sum/con_sum

    # for i in range(len(dis)):
    #     if np.isnan(dis[i]) or np.isnan(arr[i]) or dis[i]==0: #删除 数组中为nan值，或者距离为0（自身）
    #         delete.append(i)
    #         nan_count += 1
    # dis = np.delete(dis, delete) #删除对应的dis坐标
    # arr = np.delete(arr, delete) #删除对应的数据坐标
    # if len(dis) != 0:  #删除后仍然有数据，
    #     dis_sum = 0
    #     weighted_dis = 0
    #     for d_ in dis:
    #         dis_sum += (d_ ** -2) #平方和累加
    #     # print(dis_sum)
    #     for i in range(len(dis)):
    #         weighted_dis += (arr[i] * (dis[i] ** -2)) / dis_sum #加权的数值求和
    #     return weighted_dis, 1-nan_count/size  #dis 的大小裁剪后
    # else:
    #     return np.nan,0

def get_correlation_of_two_array(arr1, arr2): #UCF
    #获取两个数组的相似度
    assert len(arr1) == len(arr2), 'size of two array are not same'
    sum_ = 0
    count = 0
    for i in range(len(arr1)):
        if np.isnan(arr1[i]) or np.isnan(arr2[i]):
            pass
        else:
            sum_ += (arr1[i] - arr2[i]) ** 2
            count += 1
    if count == 0:  #两列的数组至少有一个为空
        # print("all nan values")
        return np.nan
    else: #存在数据
        if sum_ == 0:  #相同，相似性为1 #TODO 避免数据完全相同的情况
            return 1

        else:
            similarity = tanh(1 / sqrt(sum_ / count))  #均方根误差取tanh， 10 为常数项，为增大，相似度的差异。 #TODO *创新*
            return similarity

def get_UserBasedCollaborativeFiltering(matrix): #UCF
    #逐列获取每列与第一列的相似度，并使用其他列的值，求第一列中间值的相似度加权平均值
    # compute the correlation of each data
    # traverse all neighbor
    # if the neighbor size is large, then the estimated value far from the actual one
    # TODO 返回估计值，及置信度(nan值的比例)
    sim_sum = 0  # used for summing all similarity
    weighted_sum = 0  # used for recording sim*val
    middle = int(matrix.shape[0] / 2)  #中间行
    count = 0
    nan_count = 0
    for i in range(matrix.shape[1]):   #遍历全部的列

        sim = get_correlation_of_two_array(matrix[:, 0], matrix[:, i]) #计算两列矩阵的

        if np.isnan(sim) or np.isnan(matrix[middle, i]):
            nan_count += 1
        else:
            weighted_sum += sim * matrix[middle, i]
            sim_sum += sim
            count += 1

    if count == 0:
        return np.nan,0
    else:
        return weighted_sum / sim_sum, 1-nan_count/matrix.shape[1]

        # compute the correlation of each columns with first columms

# def get_selfcorrelation(series): #TODO  wrong
#     #获取一个时间窗口内，数据与中间值的相似度
#     # get the mean corelation val of each timestamp with the middle one
#     sum_ = 0
#     count = 0
#     size = len(series)  #序列的长度
#     middle = int(size / 2)
#     if np.isnan(series[middle]) or np.isnan(series).sum() == size: #序列全空，或者中间值为空
#         return np.nan
#     for i in range(size): #遍历全部series
#         if series[i] != np.nan:
#             count += 1
#             sum_ += (series[i] - series[middle]) ** 2
#     #             print(count,' ',sum_)
#     if count == 0:
#         return np.nan
#     else:
#         return sqrt(sum_ / count)



# get_selfcorrelation([np.nan,np.nan,np.nan,np.nan])
def get_ItembasedCollaborativefiltering(matrix):# ICF

    #TODO 返回估计值，及置信度(nan值的比例)
    # 比较每一行与之间行的额关联度，求加权值
    sim_sum = 0  # used for summing all similarity
    weighted_sum = 0  # used for recording sim*val
    middle = int(matrix.shape[0] / 2)
    count = 0
    nan_count = 0
    for i in range(matrix.shape[0]): #遍历列号，
        # sim = get_selfcorrelation(matrix[:, i]) #TODO wrong
        sim = get_correlation_of_two_array(matrix[i, :], matrix[middle, :])
        #         print(i,sim)
        if not np.isnan(sim) and  not np.isnan(matrix[i, 0]):  #相似度和目标值都不为空，可避开自身与其他空列
            count += 1
            sim_sum += sim
            weighted_sum += sim * matrix[i, 0]
        else:
            nan_count += 1
    #     print(count)
    #     print(sim_sum)
    if count == 0 or sim_sum == 0:
        return np.nan,0
    else:
        return weighted_sum / sim_sum, 1-nan_count/matrix.shape[0]

def getweightedNanMean(arr):
    #print(np.isnan(arr).sum())
    assert np.isnan(arr).sum() < len(arr), 'values of arr are all NANs'
    max_ = np.nanmax(arr)
    #     min_ = np.nanmin(arr)
    mean = np.nanmean(arr)   # add a random val to avoid same val
    return mean
def getweightNanMeanWithConfidence(arr,con):
    #判断nan
    assert len(arr)==len(con), 'The length of arr is not equal to confidence arr'
    sum = 0
    con_sum = 0
    for i in range(len(arr)):
        if np.isnan(arr[i]) or np.isnan(con[i]):
            pass
        else:
            sum += arr[i]*con[i]
            con_sum += con[i]
    if np.isnan(sum) or np.isnan(con_sum) or con_sum==0:
        return np.nan
    else:
        return sum/con_sum
def getweightedNanMean2(arr,weight=[0.7,0.1,0.1,0.1]):
    #print(np.isnan(arr).sum())
    assert np.isnan(arr).sum() < len(arr), 'values of arr are all NANs'
    max_ = np.nanmax(arr)
    #     min_ = np.nanmin(arr)
    assert len(arr)==len(weight),'weight is not fit for arr'
    numerator = 0 #分子
    denominator = 0 #分母
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            numerator += weight[i]*arr[i]
            denominator += weight[i]
    mean = numerator/denominator  # add a random val to avoid same val
    return mean

def get_nan_rate(series):
    assert len(series)>0, 'Error: Empty series'
    # series = series.values()
    nan_num = 0
    for i in range(len(series)):
        if np.isnan(series[i]):
            nan_num +=1
    return nan_num/len(series)

def imputebyOneID(df_clip, aq_info, aq_dis, id_, w, k, bata):  #对于缺失数据进行整体的插值
    data_df = df_clip
    imputingdf = data_df.get_df_by_id(id_)  # get df
    # print(imputingdf.head())
    dis = aq_dis.loc[id_, :][0:k].values  # get station_id neighbor distance
    indexinList = data_df.get_listIndex_by_id(id_)  # get data index in class.list
    imputed_d = 0
    list_ = []
    meanInsertedNum = 0
    for row in imputingdf.index:  # clqtest
        #         print(imputingdf.iloc[row,:])
        for index, column in enumerate(imputingdf.columns[2:], start=2):  # 0:station_id 1:utc_time 2: PM2.5 traverse the all numerical data
            if np.isnan(imputingdf.loc[row, column]):
                #                 get_dataMetrix(df_clip,neighbor_df,impute_name,columns,index,w,k)
                #                 print(imputingdf.loc[row,:])
                # print0m
                # get matrix
                # if column == 'PM2.5':
                #     print('debug')
                middle = int(w / 2)
                matrix = get_dataMetrix(data_df, aq_info, impute_name=id_, index=row, columns=index, w=w, k=k)
                #print(matrix)
                # 1
                ses_d = get_SimpleExponentialSmooth(matrix[:, 0], bata=bata)
                #print('ses_d', ses_d)
                # get dis array
                # 2
                idw_d = get_InverseDistanceWeighting(matrix[middle, :], dis)  #
                #print('idw_d', idw_d)
                # 3
                ubcf_d = get_UserBasedCollaborativeFiltering(matrix)
                #print('ubcf_d', ubcf_d)
                # 4
                ibcf_d = get_ItembasedCollaborativefiltering(matrix)
                #print('ibcf_d', ibcf_d)
                if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):
                    #print(row, column, 'insert all NAN, we replace this by mean')
                    mean_d = data_df.mean_[indexinList].loc[column]
                    imputed_d = mean_d
                    #print('imputed_d = mean', imputed_d)
                    meanInsertedNum += 1
                    #print()

                else:
                    list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
                    imputed_d = getweightedNanMean(list_)
                    #print('imputed_d != mean', imputed_d)
                    # print()
                imputingdf.loc[row, column] = imputed_d
            else:
                pass
    print('meanInsertedNum:', meanInsertedNum)
    return imputingdf
def imputebyOneID_meo(df_clip, meo_info, meo_dis, id_, w, k, bata):
    data_df = df_clip
    imputingdf = data_df.get_df_by_id(id_)  # get df
    for start in range(len(imputingdf.columns)):
        if imputingdf.columns[start] == 'temperature':
            break
    print(start)  # two meo files are different in columns
    print(imputingdf.columns[start:start+5])
    # print(imputingdf.head())
    dis = meo_dis.loc[id_, :][0:k].values  # get station_id neighbor distance
    indexinList = data_df.get_listIndex_by_id(id_)  # get data index in class.list
    # imputed_d = 0
    # list_ = []
    meanInsertedNum = 0
    # columns = ['temperature','pressure','humidity','wind_speed','wind_direction']
    for row in imputingdf.index:  # clqtest
        print(id_, row)
        if id_ == 'pinggu_meo' and row == 4965:
            print('debug')
        #         print(imputingdf.iloc[row,:])
        for index, column in enumerate(imputingdf.columns[start:start+5], start=start):  # 0:station_id 1:utc_time 2: PM2.5 traverse the all numerical data
            if np.isnan(imputingdf.loc[row, column]):
                print(imputingdf.loc[row, column])
                #                 get_dataMetrix(df_clip,neighbor_df,impute_name,columns,index,w,k)
                #                 print(imputingdf.loc[row,:])
                # print0m
                # get matrix
                # if column == 'PM2.5':
                #     print('debug')
                middle = int(w / 2)
                matrix = get_dataMetrix(data_df, meo_info, impute_name=id_, index=row, columns=index, w=w, k=k)
                print(matrix)
                # 1
                ses_d = get_SimpleExponentialSmooth(matrix[:, 0], bata=bata)
                print('ses_d', ses_d)
                # get dis array
                # 2
                idw_d = get_InverseDistanceWeighting(matrix[middle, :], dis)  #
                print('idw_d', idw_d)
                # 3
                ubcf_d = get_UserBasedCollaborativeFiltering(matrix)
                print('ubcf_d', ubcf_d)
                # 4
                ibcf_d = get_ItembasedCollaborativefiltering(matrix)
                print('ibcf_d', ibcf_d)
                if np.isnan(ses_d) and np.isnan(idw_d) and np.isnan(ubcf_d) and np.isnan(ibcf_d):
                    #print(row, column, 'insert all NAN, we replace this by mean')
                    mean_d = data_df.mean_[indexinList].loc[column]  #填充平均值
                    imputed_d = mean_d
                    #print('imputed_d = mean', imputed_d)
                    meanInsertedNum += 1
                    #print()

                else:
                    list_ = [ses_d, idw_d, ubcf_d, ibcf_d]
                    imputed_d = getweightedNanMean(list_)
                    #print('imputed_d != mean', imputed_d)
                    # print()
                imputingdf.loc[row, column] = imputed_d
            else:
                pass
    print('meanInsertedNum:', meanInsertedNum)
    return imputingdf
#导入数据 北京空气17-18污染数据
# def impute_data(fileName, neighbor_info, w = 5, k = 5):
#     #filename: data address
#     #neighbor_info: k-nearest-neighbors info for each stations
#     #w: time windows in each data imputer, the value of w should be odd
#     #k: nearest neighbors for each data imputing op
#     df = pd.read_csv(fileName)
#     aq_info = pd.read_csv(neighbor_info)
#     assert w % 2 == 1, "the value of w should be odd"
#
#     #root_dir = os.path.abspath('.')
#     #data_origindir = os.path.join(root_dir, 'KDD2018Original')
#
#         #get full data
#
#         #get each parameters
#
#         #get one time
#
#         #judge empty
#
#         # if yes, get neighbor data info
#
#         #  get_IDW
#
#         #  get_SES
#
#         #  get_UCF
#
#         #  get_ICF
def aq_fill_data():
    root_dir = os.path.abspath('../util')
    data_origindir = os.path.join(root_dir, 'AllDataset/TempData/origin') #获取文件地址
    data_savedir = os.path.join(root_dir,'AllDataset/TempData/origin')   #加载保存文件地址
    # rootdir = 'KDD2018Original'
    fileNames = ['beijing_17_18_aq',          #需要处理的文件名
                 'beijing_201802_201803_aq'
                 ]

    neighbor_info = "AllDataset/TempData/aq_aq_neighbor.csv" #邻居表格
    aq_aq_dis_info = "AllDataset/TempData/aq_aq_dis.csv"     #邻居节点距离
    for fileName in fileNames: #
        fileName_csv = fileName +'.csv'
        df = pd.read_csv(os.path.join(data_origindir,fileName_csv), index_col='Unnamed: 0')
        print(df.head())
        aq_info = pd.read_csv(neighbor_info, index_col='Unnamed: 0')  #operated data exist index colums
        print(aq_info.head())
        aq_aq_dis = pd.read_csv(aq_aq_dis_info, index_col='Unnamed: 0')
        print(aq_aq_dis.head())
        # clip dataframe by each station_id
        df_byid = df_clip(df, 'station_id')
        for id_ in df_byid.id:
            print(id_)
            imputebyOneID(df_byid, aq_info, aq_aq_dis, id_, w = 7, k = 10, bata = 0.85)
        mergeddf = df_byid.get_mergedf()
        outfilename = 'stdp' + fileName + '.csv'
        mergeddf.to_csv(os.path.join(data_savedir, outfilename))
def meo_fill_data():
    root_dir = os.path.abspath('../util')
    data_origindir = os.path.join(root_dir, 'AllDataset/TempData/origin')
    data_savedir = os.path.join(root_dir,'AllDataset/TempData/origin')
    # rootdir = 'KDD2018Original'
    fileNames = ['beijing_17_18_meo',
                 'beijing_201802_201803_me'
                 ]

    neighbor_info = "AllDataset/TempData/meo_meo_neighbor.csv"
    meo_meo_dis_info = "AllDataset/TempData/meo_meo_dis.csv"
    for fileName in fileNames:
        fileName_csv = fileName +'.csv'
        df = pd.read_csv(os.path.join(data_origindir, fileName_csv), index_col='Unnamed: 0')
        print(df.head())
        meo_neighbor = pd.read_csv(neighbor_info, index_col='Unnamed: 0')  #operated data exist index colums
        print(meo_neighbor.head())
        meo_meo_dis = pd.read_csv(meo_meo_dis_info, index_col='Unnamed: 0')
        print(meo_meo_dis.head())
        # clip dataframe by each station_id
        df_byid = df_clip(df, 'station_id')
        for id_ in df_byid.id:
            print(id_)
            if id_ == 'pingchang_meo':
                print('debug')
            imputebyOneID_meo(df_byid, meo_neighbor, meo_meo_dis, id_, w = 7, k = 10, bata = 0.85)
        mergeddf = df_byid.get_mergedf()
        outfilename = 'stdp' + fileName + '.csv'
        mergeddf.to_csv(os.path.join(data_savedir, outfilename))
if __name__ == "__main__":

    #for fileName in ["beijing_17_18_aq.csv", "beijing_201802_201803_aq.csv"]:
    # for fileName in ["beijing_201802_201803_aq.csv"]:
    #     print(fileName)
    #     impute_data(fileName, cleanModel)
    # fileNames = ['KDD2018Original/beijing_17_18_aq.csv',
    #              'KDD2018Original/beijing_201802_201803_aq.csv']
    # neighbor_info = "TempData/aq_info.csv"
    # impute_data(fileNames[0], neighbor_info, w = 5, k =5)



    #op1 impute ap data
    aq_fill_data()
    #op2 impute meo data
    meo_fill_data()