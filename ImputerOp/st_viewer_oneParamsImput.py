from utils.func import _diff
from utils.multi_view_impute_fn import *
from pathlib import Path
from sklearn import linear_model
from ImputerOp.multiobjXGBoost import MultiXGboost
import warnings
warnings.filterwarnings("ignore")
# import pickle
#TODO pickle模型存储
#  pickle.dumps(clf)
#  clf2 = pickle.loads(s)
#  clf2.predict(X[0:1])

#随机化种子设定

#TODO sklearn 模型存储与加载
import joblib
# 保存模型 joblib.dump(prams,'prams.pkl')
# 加载模型 ＃prams = joblib.load("prams.pkl")

# def insert_multi_viewer_estimate(oped_data, indexdf, item, outfile, neighbordf, neighbor_dis_df, knnVal=-1, wind=7):
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
#     with open('result/result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
#         print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
#     return mae, mre

class st_mvl():
    def __init__(self,oped_data, indexdf, item, outfilepath, neighbordf, neighbor_dis_df, knnVal=-1, wind=7,testid_size = -1):
        self.originaldata = df_clip(oped_data, 'station_id')  #数据按照station_id划分 df_clip类
        self.test_index = indexdf
        self.item = item  #使用列表
        self.columns = 0  #获取id对应的列号
        for i in range(self.originaldata.list[0].columns.size):  # 获取item对应的列号
            if self.originaldata.list[0].columns.values[i] == item:
                self.columns = i
                break
        self.saveFilePath = outfilepath
        self.neighborInfo = neighbordf
        self.neighborDis = neighbor_dis_df
        self.K = knnVal
        if self.K == -1:
            self.K = self.neighborInfo.shape[0]
        self.wind = wind
        self.datalength = self.originaldata.list[0].shape[0] #记录数据的长度
        self.test_station_size = testid_size
        if self.test_station_size == -1:
            self.test_station_size = self.neighborInfo.shape[0]

        #加权融合的权重，和偏置初始化
        self.weight = np.ones([self.neighborInfo.shape[0],4]) * 0.25 #待确定
        self.bias = np.random.random([self.neighborInfo.shape[0]])

        self.LRtraining_data = [pd.DataFrame([],columns=['idw','ses','ucf','icf','y']) for i in range(self.test_station_size)]
        self.model_list = [linear_model.LinearRegression(n_jobs=10) for i in range(self.test_station_size)]
        # self.xgboost_list = [MissingImputer(max_iter = 100, ini_fill= True, model_reg='xgboost', model_clf='xgboost')
        #                 for i in range(self.test_station_size)]  #为每个检测站点训练一个xgboost模型  #missinImputer method

        params = {'n_estimators': 360, 'colsample_bytree': 0.7166589051315236, 'gamma': 0.012954988155888664,
                  'learning_rate': 0.09067065786724686, 'max_depth': 4, 'reg_alpha': 0.35325351285730894,
                  'subsample': 0.8719641998030547}

        # self.xgboost_list = [MultiXGboost(params) for i in range(self.test_station_size)]  # 为每个检测站点训练一个xgboost模型 为每个站点设置一个模型
        # self.xgboostModel = MissingImputer(max_iter = 30, ini_fill= True, model_reg='xgboost', model_clf='xgboost')
        self.xgboostModel = MultiXGboost(params)  #使用


        self.xgboost_filled_df = df_clip(oped_data.copy(), 'station_id') #记录xgboost填充的数据
        #self.learningRate = 1e-5
        #用于存储缺失数据情况
        self.block_missing_log_time = pd.DataFrame([],columns=['station_id','index','steps']) #统计每个站点上连续时间缺失的情况
        self.block_missing_log_space = pd.DataFrame([], columns=['index']) #统计同一时刻，全体节点数据缺失的情况
        self.space_loss_time = pd.DataFrame([],columns=['index','steps'])  #统计全体数据缺失的时刻，以及连续步长
        self.check_blockmissing() #获取上面三个变量的值
        # self.check_file() #核对中间文件是否已经存储，如果存储，则加载，否则再相应程序中计算
        self.block_missing_filled_data = df_clip(oped_data.copy(), 'station_id')   #存储添加block missing的数据

    #TODO 获取原始数据缺失情况
    def construct_data_matrix_by_columnsname(self, item):
        #TODO
        # 按照item name重构data matrix
        # 返回matrix矩阵中对应位置是否为nan值
        result_df = pd.DataFrame()
        first = True
        for data_byid in self.originaldata.list:
            if first:
                result_df = data_byid[item]
                first = False
            else:
                result_df = pd.concat([result_df, data_byid[item]],axis=1)
        # print(result_df)
        result_df.columns = self.originaldata.id
        return result_df.isna()
    def check_blockmissing(self):
        #查找块缺失部分
        #重构 数据矩阵
        nan_matrix = self.construct_data_matrix_by_columnsname(self.item)
        #TODO 1 对nan_matrix 做行统计，做列统计。
        # 大于window数量的缺失为时间block缺失
        block_missing_file_time = self.saveFilePath + 'block_missing_log_time.csv'
        block_missing_file_time_path = Path(block_missing_file_time)
        file1_exsit = False

        block_missing_file_space = self.saveFilePath + 'block_missing_log_space.csv'
        block_missing_file_space_path = Path(block_missing_file_space)
        file2_exsit = False

        space_loss_time = self.saveFilePath + 'space_missing_time.csv'
        space_loss_time_path = Path(space_loss_time)
        file3_exsit = False

        if block_missing_file_time_path.exists():
            file1_exsit = True
        if block_missing_file_space_path.exists():
            file2_exsit = True
        if space_loss_time_path.exists():
            file3_exsit = True

        if not file1_exsit:
            for id in nan_matrix.columns:
                start = 0
                steps = 0
                first_nan = True # first_nan = True, steps =0; first_nan = False, steps >1
                for row in nan_matrix.index:
                    assert (first_nan == True and steps == 0) or (first_nan == False and steps >0), "Lock fails..."
                    if nan_matrix.loc[row,id]: #TODO 1当前数据有缺失
                        if first_nan:  #TODO 1.1第一次进入block区域
                            start = row
                            steps = 1
                            first_nan = False
                        else: #TODO 1.2第2+进入缺失区域，步长+1
                            steps += 1 #连续缺失，步长加1
                    else: #TODO 2当前无缺失值
                        if steps > 0: #TODO 2.1缺失block结束
                            # block模块结束
                            # 记录之前的信息
                            series = pd.Series([id, start, steps], index=['station_id', 'index', 'steps'])
                            self.block_missing_log_time = self.block_missing_log_time.append(series, ignore_index=True)
                            first_nan = True
                            steps = 0
                        else: #TODO 2.2处于无缺失区域
                            # 处于无缺失区域
                            pass
                if steps >0: # 添加末尾的数据
                    series = pd.Series([id, start, steps], index=['station_id', 'index', 'steps'])
                    self.block_missing_log_time = self.block_missing_log_time.append(series, ignore_index=True)
            self.block_missing_log_time.to_csv(block_missing_file_time)
                    #check loss
        else:
            self.block_missing_log_time = pd.read_csv(block_missing_file_time,index_col='Unnamed: 0')
        #TODO 2
        # 同一时刻，全部节点数据缺失为空间block缺失
        if not file2_exsit:
            index = []
            count_list = []
            count = 0
            for row in nan_matrix.index:
                if (nan_matrix.iloc[row,:]==True).all():
                    index.append(row)
            # print(index)
            self.block_missing_log_space['index'] = index
            self.block_missing_log_space.to_csv(block_missing_file_space)
            #check row empty
        else:
            self.block_missing_log_space = pd.read_csv(block_missing_file_space, index_col='Unnamed: 0')
        # print(self.block_missing_log_space.head())
        #TODO 3 统计空间块缺失的时间连续性
        if not file3_exsit:

            temp_df = self.block_missing_log_space
            temp_df['diff'] = self.block_missing_log_space.diff()
            start = 0
            steps = 0
            # print(temp_df.head())
            # print(temp_df.loc[0,'index'])
            for row in temp_df.index:
                if np.isnan(temp_df.loc[row,'diff']): #TODO 第一行为空
                    start = temp_df.loc[row,'index']
                    steps = 1
                    first_nan = False
                else:
                    if temp_df.loc[row,'diff']>1: #第一次进入，diff>1
                        #存储上一区域的信息
                        series = pd.Series([start, steps], index=['index', 'steps'])
                        self.space_loss_time = self.space_loss_time.append(series, ignore_index=True)

                        start = temp_df.loc[row,'index']
                        steps = 1

                    else:
                        steps += 1

            #保存最后一条数据
            series = pd.Series([start, steps], index=['index', 'steps'])
            self.space_loss_time = self.space_loss_time.append(series, ignore_index=True)
            self.space_loss_time.to_csv(space_loss_time)
        else: #文件已经存在，直接加载
            self.space_loss_time = pd.read_csv(space_loss_time, index_col='Unnamed: 0')


        # print('test')
        # for data_byid in self.originaldata.list:
        #     # print(data_byid)
        #     start = 0
        #     step = 0
        #     in_block_area = False
        #     nan_list = data_byid[self.item].isna()
        #     print(nan_list)
            # for row in data_byid.index:
            #     if
            #     if row %2000==0:
            #         #核对超过wind大小的数据缺失
            #         print(row)

        #对于块缺失数据，进行指数平滑，逆距离加权
    # def op_block_missing(self):
    #     #
    #     pass
    #TODO 获取插值的数据矩阵
    def _get_data_matrix_index(self,query_row,rowSize,intervals,type):
        #TODO 获取插值矩阵的index
        intervals_ = 1
        # TODO 根据数据的类型，构造时间窗口
        if intervals == 'hour':
            intervals_ = 1
        elif intervals == 'day':
            intervals_ = 24
        elif intervals == 'week':
            intervals_ = 24 * 7
        else:
            intervals_ = 1

        # TODO 根据middle类型设置带估计值位于表格的中间或者末尾
        matrix_begin_index = 0
        matrix_end_index = 0
        index = []
        #TODO method1
        if type =='middle':
            index = [query_row + (i-int(self.wind/2))* intervals_ for i in range(self.wind)]
            target_index = int(self.wind/2)
        else:
            index = [query_row - (self.wind-1 -i) * intervals_ for i in range(self.wind)]
            target_index = self.wind
        # print(index)
        #TODO delete < 0 的index
        filtered_list = [i for i in index if i>=0 and i < rowSize]
        target_index = target_index - (self.wind - len(filtered_list))
        return filtered_list, target_index
        #TODO methods 2
        # if type == 'middle' and intervals_ == 1:  # 待估计值放置在中间, interval_取1
        #     middle_index = query_row  # 将查询行号，放置在中间
        #     half_wind_size = int(self.wind / 2)
        #     if middle_index < half_wind_size:  # 判断下边界
        #         middle_index = half_wind_size  # 数据的开始
        #     elif middle_index > self.datalength - 1 - half_wind_size:  # 判断数据的结尾
        #         middle_index = self.datalength - 1 - half_wind_size
        #     begin = int(middle_index - (self.wind - 1) / 2)  # 获取时间窗口的两侧
        #     end = int(middle_index + (self.wind - 1) / 2)  #
        # else:  # 待估计值放置在末尾
        #     end_row = query_row
        #     if query_row < self.wind * intervals_ - 1:  # 判断下边界
        #         end_row = self.wind - 1  # 数据的开始
        #     elif query_row > self.datalength - 1:  # 判断数据的结尾
        #         end_row = self.datalength - 1
        #     begin = int(end_row - (self.wind - 1))  # 获取时间窗口的两侧
        #     end = int(end_row)  #
        #     # index = query_row
        #     # while query_row - intervals_ >=0:
        #
        # print('begin', begin, 'end', end)

        #求取matrix的index值
        # index_list = []
    def get_data_matrix(self, station_id, query_row, intervals='hour',type = 'middle'):
        #TODO 返回插值矩阵的index
        index,target_index = self._get_data_matrix_index(query_row,self.datalength,intervals,type)
        #index 索引列表，target_index 目标插入时刻数据所在行
        firstLine = True  # 输出数据时，判断空操作
        matrix = pd.DataFrame([])  # 存储领域矩阵
        neighborlist = [] # 记录数据记录的index顺序
        alternative_list = [] #记录替补领域索引 ，两个索引列表主要用于提取坐标信息
        alternative_matrix = pd.DataFrame([])  # 储存数据量较少的列，当matrix矩阵未满的适合，填充alternative_matrix的前N列
        neighbor_l = self.neighborInfo.loc[station_id, :][0: 2 * self.K]  # 获取neighbor_l 时间区域的2k个列的信息
        for i, id_ in enumerate(neighbor_l, start=0):  # 遍历领域的id名
            # neighbor_index.append(df_clip.get_listIndex_by_id(n))
            # print(neighbor_index)
            # print(i)
            list_index = self.originaldata.get_listIndex_by_id(id_)
            series = self.originaldata.list[list_index].iloc[index, self.columns:self.columns + 1].copy()
            nan_num = series.isna().sum().values[0]  # 该列nan值的数量
            nan_rate = nan_num / len(series)  # nan值的比例

            if firstLine:  # 第一列插入
                matrix = series
                neighborlist.append(id_)  # 记录数据记录的index顺序
                # print(matrix)
                firstLine = False
            elif not firstLine and nan_rate < 1:  # 其他列插入，且nan值比例小于1,即非空
                matrix = pd.concat([matrix, series], axis=1)  # 铸列添加
                neighborlist.append(id_)  # 记录数据记录的index顺序
            else:  # rate= 1 全空情况，添加数据到备选矩阵
                if alternative_matrix.shape[1] == 0:
                    alternative_matrix = series
                else:
                    alternative_matrix = pd.concat([alternative_matrix, series], axis=1)  # 逐列添加
                alternative_list.append(id_)  # 按照neighbor_info 填充neighbor信息
            if matrix.shape[1] == self.K:  #TODO 提前判断结束情况
                break
        if matrix.shape[1] < self.K:  #TODO 判断matrix是否已经填满
            required_col_num = self.K - matrix.shape[1]
            matrix = pd.concat([matrix, alternative_matrix.iloc[:, :required_col_num]], axis=1)  # 按列拼接
            neighborlist.extend(alternative_list[:required_col_num])  # 拼接列表
        #根据matrix_list对应的index取，距离值
        dis_all = self.neighborDis.loc[station_id, :]  # get station_id neighbor distance
        dis = dis_all[neighborlist]
        return matrix.values,target_index, dis
        # return 0

    #TODO 多视图插值方法
    def IDW(self,data,dis_info,target_index,pow = 2):
        #TODO data:插值矩阵，dis_info：距离列表
        # target_index: 目标数据的插入行位置
        # type: 目标数据的位置类型中间或则末尾(无用处)
        # Inverse Distance Weighting
        assert data.shape[1] == len(dis_info), 'check data matrix and distance list'
        series = data[target_index, :]
        dis = np.array(dis_info)  # dis数据转化为np.array,方便**-2
        arr = np.array(series)
        # delete = []
        con_sum = 0 #记录全部的逆距离权重和
        dis_sum = 0 #记录非空数据的逆距离权重和
        weighted_sum = 0  # 加权的数值求和
        for i in range(len(dis)):
            if dis[i] == 0:  # 删除 数组中为nan值，或者距离为0（自身）
                pass
            elif np.isnan(arr[i]):  # 删除数组中为nan值（自身）
                # print('dis:',dis[i])
                # print('dis**-pow:', dis[i] ** -pow)
                con_sum += dis[i] ** -pow  # 只统计全部数值的权重和
            else:
                # print('dis:', dis[i])
                # print('dis**-pow:', dis[i] ** -pow)
                dis_sum += dis[i] ** -pow
                con_sum += dis[i] ** -pow
                weighted_sum += (arr[i] * (dis[i] ** -pow))
        if dis_sum == 0:
            return np.nan, 0
        else:
            return weighted_sum / dis_sum, dis_sum / con_sum  # 返回估计值，和缺失值置信值
        # estimate_val = 0
        # if type == 'middle':
        #     series = data[target_index,:]
        #     dis = np.array(dis_info)  #dis数据转化为np.array,方便**-2
        #     arr = np.array(series)
        #     delete = []
        #     con_sum = 0
        #     dis_sum = 0
        #     weighted_sum = 0  # 加权的数值求和
        #     for i in range(len(dis)):
        #         if dis[i] == 0:  # 删除 数组中为nan值，或者距离为0（自身）
        #             pass
        #         elif np.isnan(arr[i]):  # 删除 数组中为nan值，或者距离为0（自身）
        #             con_sum += dis[i] ** -pow #统计全部数值的权重和
        #         else:
        #             dis_sum += dis[i] ** -pow
        #             con_sum += dis[i] ** -pow
        #             weighted_sum += (arr[i] * (dis[i] ** -pow))
        #     if dis_sum == 0:
        #         return np.nan, 0
        #     else:
        #         return weighted_sum / dis_sum, dis_sum / con_sum #返回估计值，和缺失值置信值
        # else:
        #     series =  data.iloc[target_index,:]
        #     dis = np.array(dis_info)
        #     arr = np.array(series)
        #     pass
        # return estimate_val
    def SES(self,data,target_index,bata=0.8):
        #TODO Simple Exponential Smoothing (SES),
        # 获取series 第一列为待估计数据时间区域内的数据
        series = data[:,0] #获取数据列
        if np.isnan(series).all():  # 数据全为nan,
            # print('all nan')
            return np.nan, 0
        else:
            #  print('not all nan')
            size = len(series)  # series的长度
            # target_index 目标位置
            numerator = 0  # 分子
            denominator = 0  # 分母
            # nan_count = 0
            con_sum = 0

            for i in range(size):
                temp = bata * (1 - bata) ** (abs(i - target_index) - 1)  # 当i-middle时，幂值为-1，其他数据与中间间隔个数-1
                if i != target_index:  # 排除待插入位置自身的权重
                    con_sum += temp  # 计算总的权重
                if not np.isnan(series[i]) and i != target_index: #TODO 当进行插值时，目标位置为nan值，训练时也无需统计目标位置值
                    denominator += temp #分母 权重值累加
                    numerator += temp * series[i] #分子
            if denominator == 0:  #
                return np.nan, 0
            else:
                # return numerator / denominator, 1 - nan_count/size #返回比例值不合适
                return numerator / denominator, denominator / con_sum  # 权重比例用于计算可信度

        #TODO 模式一： middle
        # if type == 'middle':
        #     if np.isnan(series).all():
        #         return np.nan, 0
        #     else:
        #         size = len(series)  # series的长度
        #         middle = int(size / 2)  # 中间位置
        #         numerator = 0  # 分子
        #         denominator = 0  # 分母
        #         # nan_count = 0
        #         con_sum = 0
        #
        #         for i in range(size):
        #             temp = bata * (1 - bata) ** (abs(i - middle) - 1)  # 当i-middle时，幂值为-1，其他数据与中间间隔个数-1
        #             if i != middle:  # 排除待插入位置自身的权重
        #                 con_sum += temp  # 计算总的权重
        #             if not np.isnan(series[i]):
        #                 #    nan_count += 1
        #                 # else:
        #                 #                 print('temp',temp)
        #                 denominator += temp
        #                 #                 print('denominator',denominator)
        #                 numerator += temp * series[i]
        #
        #         #                 print('numerator',numerator)
        #         #         print('Estimator data:',numerator/denominator)
        #         if denominator == 0:  #
        #             return np.nan, 0
        #         else:
        #             # return numerator / denominator, 1 - nan_count/size #返回比例值不合适
        #             return numerator / denominator, denominator / con_sum  # 权重比例用于计算可信度
        # else:
        #     pass
    def SES_Residual(self,data,target_index,bata=0.8, decay_factor=0.01):
        # TODO Simple Exponential Smoothing (SES),
        # 获取series 第一列为待估计数据时间区域内的数据
        series = np.array(data[:, 0])  # 获取数据列
        if np.isnan(series).all():  # 数据全为nan
            # print('all nan')
            return np.nan, 0
        else:
            #计算残差
            diff = _diff(series)
            nan_index = [i for i in range(len(diff)) if np.isnan(diff[i])]
            diff[nan_index] = 0
            #  print('not all nan')
            size = len(series)  # series的长度
            # target_index 目标位置
            numerator = 0  # 分子
            denominator = 0  # 分母
            # nan_count = 0
            con_sum = 0

            for i in range(size):
                temp = bata * (1 - bata) ** (abs(i - target_index) - 1)  # 当i-middle时，幂值为-1，其他数据与中间间隔个数-1
                if i != target_index:  # 排除待插入位置自身的权重
                    con_sum += temp  # 计算总的权重
                if not np.isnan(series[i]) and i != target_index:  # TODO 当进行插值时，目标位置为nan值，训练时也无需统计目标位置值
                    denominator += temp  # 分母 权重值累加
                    numerator += temp * (series[i]+diff[i]*decay_factor)  # 分子
            if denominator == 0:  #
                return np.nan, 0
            else:
                # return numerator / denominator, 1 - nan_count/size #返回比例值不合适
                return numerator / denominator, denominator / con_sum  # 权重比例用于计算可信度
    def _get_correlation_of_two_array(self,arr1, arr2): #UCF
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
            else:  #一般情况
                #similarity_ = tanh(1 / sqrt(sum_ / count))  #均方根误差取tanh， 10 为常数项，为增大，相似度的差异。 #TODO *创新*
                similarity = 1 / sqrt(sum_ / count) # 均方根误 10 为常数项，为增大，相似度的差异。 偏差累计越小，相似度越高
                return similarity
    def ICF(self,data,target_index,type='middle'):
        #TODO  Item-based Collaborative filtering (ICF).
        # 比较每一行与之间行的额关联度，求加权值
        sim_sum = 0  # used for summing all similarity
        weighted_sum = 0  # used for recording sim*val
        #middle = int(data.shape[0] / 2)
        count = 0
        nan_count = 0
        for i in range(data.shape[0]):  # 遍历列号，
            if i!= target_index:
                sim = self._get_correlation_of_two_array(data[i, :], data[target_index, :])
                #         print(i,sim)
                if not np.isnan(sim) and not np.isnan(data[i, 0]):  # 相似度和目标值都不为空，可避开自身与其他空列
                    count += 1
                    sim_sum += sim
                    weighted_sum += sim * data[i, 0]
                else:
                    nan_count += 1
        if count == 0 or sim_sum == 0:
            return np.nan, 0
        else:
            return weighted_sum / sim_sum, 1 - nan_count / data.shape[0]
        # if type == 'middle':
        #     pass
        #
        # else:
        #     pass
    def UCF(self,data,target_index):
        #TODO User-based  Collaborative filtering (UCF)
        # 逐列获取每列与第一列的相似度，并使用其他列的值，求第一列中间值的相似度加权平均值
        sim_sum = 0  # used for summing all similarity 统计总的相似度
        weighted_sum = 0  # used for recording sim*val #加权求和值
        #middle = int(matrix.shape[0] / 2)  # 中间行
        count = 0
        nan_count = 0
        for i in range(1,data.shape[1]):  # 遍历除第一列（i=0）外的全部的列
            sim = self._get_correlation_of_two_array(data[:, 0], data[:, i])  # 计算两列矩阵的
            if np.isnan(sim) or np.isnan(data[target_index, i]): #如果相似度为空，或者候选列的对应时刻的值为空
                nan_count += 1
            else:
                # print(sim)
                # print(data[target_index, i])
                # print(sim * data[target_index, i])
                weighted_sum += sim * data[target_index, i]
                sim_sum += sim
                count += 1

        if count == 0:
            return np.nan, 0
        else:
            return weighted_sum / sim_sum, 1 - nan_count / data.shape[1]
        # if type == 'middle':
        #     pass
        # else:
        #     pass
    #TODO 获取四个估计值
    def _get_four_estimation(self,station_id,row, intervals='hour', type='middle'):
        #TODO 使用四种估计模型，估计某一时刻
        # 输入参数：
        data_matrix, target_index, neighbor_dis = self.get_data_matrix(station_id, row, intervals=intervals, type=type)
        # TODO 四种估计方法，
        # 返回四种方法的估计值数组，以及置信值数组
        # print(data_matrix)
        # print(neighbor_dis)
        idw_d, idw_conf = self.IDW(data_matrix, neighbor_dis, target_index)
        # ses_d, ses_conf = self.SES(data_matrix, target_index)
        ses_d, ses_conf = self.SES_Residual(data_matrix, target_index)
        ucf_d, ucf_conf = self.UCF(data_matrix, target_index)
        icf_d, icf_conf = self.ICF(data_matrix, target_index)
        if np.isnan([idw_d,ses_d,ucf_d,icf_d]).any():
            print('debug')
        return [idw_d,ses_d,ucf_d,icf_d],[idw_conf,ses_conf,ucf_conf,icf_conf]
    #TODO 获取LR模型训练数据
    def get_LinearR_trainingdata(self,station_id,intervals='hour', type='middle'):
        #TODO 该函数实现对于某一站点的数据，进行无缺失部分的提取，线性回归模型的训练
        # 原始数据存储位置 self.originaldata.list[i]，处理后数据存储位置
        # self.LRtraining_data[i]中
        listIndex = self.originaldata.get_listIndex_by_id(station_id)
        if self.LRtraining_data[listIndex].shape[0] >0:  #判断数据是否已经存在
            return self.LRtraining_data[listIndex]
        else:
            outfile_name = self.saveFilePath + station_id + 'trainLRdata.csv'
            outfile_name_path = Path(outfile_name)
            #判断数据文件是否存在
            if outfile_name_path.exists(): #如果文件存在则加载该文件，减少计算时间
                self.LRtraining_data[listIndex] = pd.read_csv(outfile_name_path)
                print(station_id,'loading training data')

            else: #否则计算存储文件
                #计算数据
                # 获取存储键值
                for row in self.originaldata.list[listIndex].index: #
                    #求取训练集：
                    esti_list, conf_list = self._get_four_estimation(station_id, row)
                    esti_list.append(self.originaldata.list[listIndex].loc[row,self.item])  #添加真实值
                    series = pd.Series(esti_list, index=['idw','ses','ucf','icf','y'])
                    self.LRtraining_data[listIndex] = \
                        self.LRtraining_data[listIndex].append(series, ignore_index=True)
                #拼接数据
                # outfile_name = self.saveFilePath + station_id +'trainLRdata.csv'
                self.LRtraining_data[listIndex].to_csv(outfile_name)
        #存储训练集
    def weight_sum(self,arr,station_index):
        #TODO arr:估计值[idw,ses,ucf,icf]
        assert len(arr) == self.weight.shape[1],'The length of arr is not fit the weight array'
        sum = self.bias[station_index]
        for i,val_ in enumerate(arr, start=0):
            sum += self.weight[station_index,i] * val_
        return sum
    #TODO 手动更新权重计算[update_weight_bias,train_weight],未完整
    # def update_weight_bias(self,delta,input_arr,target_index):
    #     #TODO 权重更新过程 batch_learning,
    #     temp = self.learningRate * delta
    #     for i, x in enumerate(input_arr,start=0):
    #         self.weight[target_index,i] -= temp * input_arr[i]
    #     self.bias[target_index] -= temp
    # def train_weight(self):
    #     #TODO 通过已有数据集训练加权模型的权重，偏置
    #     # for i in range(self.test_station_size):
    #     # 添加判断参数文件（weight bias）是否存在，如果存在则加载。
    #     # 否则重新训练
    #     for i in range(1):
    #         station_df = self.originaldata.list[i]  #获取数据
    #         station_id = self.originaldata.id[i]    #获取id
    #
    #         for row in station_df.index:
    #             if row%1000 ==0:
    #                 print(row)
    #             if not np.isnan(station_df.loc[row,self.item]):
    #                 data_matrix,target_index,neighbor_dis = self.get_data_matrix(station_id,row,intervals='hour', type = 'middle')
    #                 #TODO 四种估计方法，
    #                 # print(data_matrix)
    #                 # print(neighbor_dis)
    #                 idw_d, idw_conf = self.IDW(data_matrix, neighbor_dis, target_index)
    #                 # ses_d, ses_conf = self.SES(data_matrix, target_index)
    #                 ses_d, ses_conf = self.SES_Residual(data_matrix, target_index)
    #                 ucf_d, ucf_conf = self.UCF(data_matrix, target_index)
    #                 icf_d, icf_conf = self.ICF(data_matrix, target_index)
    #                 #TODO 求估计值[idw,ses,ucf,icf]
    #                 input = [idw_d,ses_d,ucf_d,icf_d]
    #                 y_est = self.weight_sum([idw_d,ses_d,ucf_d,icf_d],i)
    #                 #TODO 计算偏差值 ，求梯度值
    #                 # y_est = sum(w(i)*x(i)) + b(i)
    #                 # Loss = (y_est - data_matrix[target_index,0])**2 /2  #梯度 delta(w（i）)=（y_est - data_matrix[target_index,0]）* x(i)
    #                 delta = y_est - data_matrix[target_index,0]
    #                 #TODO 反向传播
    #                 self.update_weight_bias(delta,input,i)
    #                 # print('test')
    #             else:
    #                 pass
    #
    #         #    if row not in self.block_missing_log_space:

    #TODO 使用sklearn计算回归模型
    #TODO 两个学习的模型
    #TODO 训练LR模型
    def train_LinearRegression_model(self):
        # TODO 通过已有数据集训练加权模型的权重，偏置
        # for i in range(self.test_station_size):
        # 添加判断参数文件（weight bias）是否存在，如果存在则加载。
        # 否则重新训练
        for i in range(self.test_station_size):    #遍历station站点
            # station_df = self.originaldata.list[i]  # 获取数据
            station_id = self.originaldata.id[i]  # 获取id
            model_parms_ = self.saveFilePath + station_id + '_LR_.pkl'
            model_parms_path = Path(model_parms_)
            if model_parms_path.exists():   #
                print('prams params file exist')
                # 加载模型 ＃prams = joblib.load("prams.pkl")
                self.model_list[i] = joblib.load(model_parms_path)
                #test load model
                # self.get_LinearR_trainingdata(station_id)  #
                # sub_df = self.LRtraining_data[i].copy()
                # sub_df = sub_df.dropna()
                # # 分割输入数据，和标签数据
                # input_d = sub_df[['idw', 'ses', 'ucf', 'icf']].values
                # estimate = self.model_list[i].predict(input_d)
                # out = pd.DataFrame(estimate)
                # out.to_csv('result2.csv')
            else: #
                print('prams params not exist')
                #模型文件不存在
                #TODO 模型 self.model_list[0]
                # 提取训练数据
                self.get_LinearR_trainingdata(station_id) #
                #训练模型
                #获取训练数据，删除有nan值的行
                sub_df = self.LRtraining_data[i].copy()
                sub_df = sub_df.dropna()
                #分割输入数据，和标签数据
                input_d = sub_df[['idw','ses','ucf','icf']].values
                y_d = sub_df[['y']].values
                #训练模型
                self.model_list[i].fit(input_d,y_d)
                ## 保存模型 joblib.dump(prams,'prams.pkl')
                joblib.dump(self.model_list[i], model_parms_path)
                # print(self.model_list[i].coef_)
                #TODO 预测一个值使用
                # estimate = self.model_list[i].predict(input_d)

                #TODO test reading
                # testmodel = joblib.load(model_parms_path)
                # estimate2 = testmodel.predict(input_d)
                # delta = abs(estimate-estimate2)
                #TODO write estimate
                # delta = abs(estimate - y_d)
                # print(delta)
                # df_ = pd.DataFrame([],columns = ['estimate','real','delta'])
                # df_['estimate'] = estimate.reshape(-1)
                # df_['real'] = y_d.reshape(-1)
                # df_['delta'] = delta.reshape(-1)
                # df_.to_csv('result.csv')
    # def _xgb_objective_fn(self,params_space):
    #     #TODO xgboost模型参数优化算法
    #     # params_space :dic词典型变量，用于记录参数的选择空间。
    #     # 使用全部数据作为训练集，使用挖出的数据作为测试集，寻找最优的参数
    #
    #     model =  xgb.XGBRegressor(
    #         max_depth = params_space['max_depth'],
    #         n_estimators = int(params_space['n_estimators']),
    #         subsample = params_space['subsample'],
    #         colsample_bytree = params_space['colsample_bytree'],
    #         learning_rate = params_space['learning_rate'],
    #         reg_alpha = params_space['reg_alpha']
    #     )
        #模型训练并且估计

        #

        #最终填充的数据，与挖出的数据求rmse


    # def _xgboost_params_opt(self,trainX, trainY):
    #     #TODO 本函数用于训练一个xgboost模型，
    #     # 使用贝叶斯模型调参。
    #     all_train_df = self.originaldata.get_mergedf().copy()
    #     train_df = all_train_df.dropna(axis=0, how='any').reset_index(drop=True)
    #     print(train_df.iloc[:, -6:])
    #     # self.xgboostModel.fit(train_df.iloc[:,-6:])  #
    #     self.xgboostModel.fit(all_train_df.iloc[:, -6:])  #
    #     joblib.dump(self.xgboostModel, self.saveFilePath + "xgb.model.dat")
    #     # test
    #     xgb_ = joblib.load(self.saveFilePath + "xgb.model.dat")
    #
    #     filled_df = xgb_.transform(all_train_df.iloc[:, -6:])
    #     filled_df.to_csv(self.saveFilePath + '1.csv')

    def train_xgboost_model(self):
        #TODO 存在问题，估计值存在负数
        #TODO 两种策略：2：全部数据训练一个模型
        #train one xgboost
        # df_ = self.originaldata.get_mergedf().copy()
        # df_2 = df_.dropna(axis=0,how='any').reset_index(drop=True)
        # numerialMatric = df_2.iloc[:, -6:]  # 获取删除测试数据后的数据区域
        # self.xgboost_list[0].fit(numerialMatric)
        # df_.iloc[:, -6:] = self.xgboost_list[0].transform(df_.iloc[:, -6:])
        # df_.to_csv(self.saveFilePath+'xgboost.csv')

        #整理训练集，训练模型 self.xgboostModel。
        model_params = self.saveFilePath+"xgb.model.dat"
        model_params_path = Path(model_params)
        if model_params_path.exists():
            # TODO 模型存在，则加载模型
            self.xgboostModel = joblib.load(model_params_path)
            # all_train_df = self.originaldata.get_mergedf().copy()
            # filled_df = self.xgboostModel.transform(all_train_df.iloc[:, -6:])
            # all_train_df.iloc[:, -6:] = filled_df
            # all_train_df.to_csv(self.saveFilePath + '1.csv')
        else:
            # TODO 否则则训练模型
            all_train_df = self.originaldata.get_mergedf().copy()
            train_df = all_train_df.dropna(axis=0,how='any').reset_index(drop=True)
            print(train_df.iloc[:,-6:])
            # self.xgboostModel.fit(train_df.iloc[:,-6:])  #
            self.xgboostModel.fit(all_train_df.iloc[:,-6:])  #
            joblib.dump(self.xgboostModel,self.saveFilePath+"xgb.model.dat")
            # test
            # xgb = joblib.load(model_params)
            # filled_df = xgb.transform(all_train_df.iloc[:,-6:])
            # all_train_df.iloc[:,-6:] = filled_df
            # all_train_df.to_csv(self.saveFilePath+'1.csv')



        # TODO 两种策略：1：每个站点训练一个模型
        # for i in range(self.test_station_size):    #遍历station站点
        #     station_id = self.originaldata.id[i]  # 获取id
        #     model_parms_ = self.saveFilePath + station_id + 'XGboostparam.xgboost.pkl'
        #     model_params_path = Path(model_parms_)
        #     # if model_params_path.exists():
        #     if not model_params_path.exists():  # debug
        #         print(station_id,'XGboost prams params file exist')
        #         # 加载模型 ＃prams = joblib.load("prams.pkl")
        #         self.xgboost_list[i] = joblib.load(model_parms_path)
        #         #TODO 测试xgboost模型的加载
        #         # sub_df = self.originaldata.list[i].copy()
        #         # print(sub_df.iloc[6, -6:].values.reshape(1, -1))
        #         # input = sub_df.iloc[6, -6:].values.reshape(1, -1)
        #         # estimate = self.xgboost_list[i].transform(input)
        #     else:
        #         print(station_id,'prams params not exist')
        #         sub_df = self.originaldata.list[i].copy()
        #         numerialMatric = sub_df.iloc[:, -6:]  # 获取删除测试数据后的数据区域
        #         self.xgboost_list[i].fit(numerialMatric)
        #         self.xgboost_filled_df.list[i].iloc[:,-6:] = self.xgboost_list[i].transform(numerialMatric)
        #         self.xgboost_filled_df.list[i].to_csv(self.saveFilePath+str(i)+'xgboost_filled.csv')
        #         joblib.dump(self.xgboost_list[i],model_parms_path)
        #         #TODO 使用xgboost，预测一个值
        #         # estimate = self.xgboost_list[i].transform(sub_df.iloc[0:6, -6:] )
        #         # print('debug')
        # out_df = self.xgboost_filled_df.get_mergedf()
        # out_df.to_csv(self.saveFilePath+'xgboost_filled.csv')
    def check_nan(self,arr):
        #TODO 核对数组是否全为 nan
        notNanList = []
        Full = True
        for i in range(arr):
            if np.isnan(arr[i]): #存在nan值，则数据不完整，Full为False
                Full = False
            else:
                notNanList.append(i) #存储非空位置index
        return notNanList, Full
    def  _blockArea_missing_analyze(self,station_id, row, step, w, intervals='hour', type='middle'):
        #TODO 核对空间上数据是否全部缺失，查看 self.block_missing_log_space
        # get block missing area local:获取当前时刻本地数据是否全部趋势
        # get block missing area global：获取当前时刻全部节点的数据是否缺失。
        # 返回对于各step数据的插值方法。
        # 不考虑 【邻居节点数据未缺失，他数据属性未缺失---> IDW+SES； 邻居节点数据未缺失，其他数据属性缺失---> IDW，多视图插值总会成功】
        # 0:邻居节点数据缺失,其他属性未缺失---> K近邻的插值 或者xgboost
        # 1:邻居节点数据缺失，其他属性缺失--->丢弃数据
        # output: True 使用xgboost估计，， False 不适用xgboost估计[case1 相邻节点有数据可以估计 case2 大量数据丢失，直接删除]
        #insert_index = []

        #TODO 1005xgboost使用条件
        # 使用条件：同一时刻，待估计属性未完全缺失，
        # 使用情况：1：同时刻其他站点该属性完全缺失 2：属性邻近发生连续缺失

        if row in self.block_missing_log_space['index'].values or step > w: #判断row是否在全部缺失数据中
            #该时间点，全部节点数据缺失
            #TODO check 本地时刻其他属性数据是否全部缺失
            list_index = self.originaldata.get_listIndex_by_id(station_id) #获取index
            print(self.originaldata.list[list_index].loc[row,['PM2.5','PM10','NO2','CO','O3','SO2']])
            loss_all = self.originaldata.list[list_index].loc[row,['PM2.5','PM10','NO2','CO','O3','SO2']].isna().all() #获取该时间，节点数据
            # if loss_all: #此时全部数据为空
            #     #insert_index.append(0)  #丢弃该数据
            #     return
            # else:  #此时数据不全为空
            #     #gxboost,或者knn
            #     insert_index.append(1)
            # print('test')
            return not loss_all #全部缺失则返回 False, #没有全部缺失则返回True
        else:
            return False

    def _check_index_in_blocking_missing(self,station_id,check_row,w):
        #TODO 该函数用来检测某一行，是否存在block missing区域
        #返回： 是否为block missing 区域数据：flag
        search_df = self.space_loss_time.copy()
        search_df = search_df.sort_values(by='index')
        flag = False
        for row in search_df.index:
            start_index = search_df.loc[row,'index']
            end_index = search_df.loc[row,'index'] + search_df.loc[row,'steps'] -1
            if check_row >= start_index and check_row <=end_index:
                if search_df.loc[row,'steps'] >w:  #TODO 如果缺失数据过多，则返回丢弃标志
                    flag = True
                break
            if check_row < start_index:
                break
        return flag
    #TODO 数据插值操作
    def _block_missing_op(self,intervals='hour', type='middle'):
        block_missing_data_filled_filename = self.saveFilePath + 'block_missing_data_filled_file.csv'
        block_missing_data_filled_filename_path = Path(block_missing_data_filled_filename)
        #TODO 为加快调试速度，处理文件将存储在本地，可直接加载
        if block_missing_data_filled_filename_path.exists():  # debug 模式，为避免重复计算，暂存计算中间量。如果block缺失数据已经存储，直接加载，当参数修改后，文件删除，重新计算
            block_missing_dara_filled_df = pd.read_csv(block_missing_data_filled_filename_path, index_col='Unnamed: 0')
            self.block_missing_filled_data = df_clip(block_missing_dara_filled_df, 'station_id')
            self.originaldata = df_clip(block_missing_dara_filled_df.copy(), 'station_id')
        else:
            #TODO 对于block_missing_log_time 按照进行index排序
            tempdf = self.block_missing_log_time.copy().sort_values(by='index')
            tempdf = tempdf.reset_index(drop=True)  # 对于missing log数据按照原始数据的index排序，重新设置df的index
            # print(tempdf.head())
            for row in tempdf.index:  #TODO 时间维度上，遍历全部缺失的数据
                if(row%1==0):
                    print('Block Missing Running process: ',row,'/',len(tempdf))
                # if tempdf.loc[row, 'steps'] >= self.wind:  # 对于时间上，大于缺失插值窗口大小的数据进行预先填充, 改为对于时许缺失数据全部全部分析
                #TODO 对于数据进行插值
                if tempdf.loc[row, 'steps'] >= 1:  #
                    # print(tempdf.loc[row,'index'])
                    # 对于连续的数据缺失填充 idw ses
                    start_index = tempdf.loc[row, 'index'] #获得原始数据所在行
                    #TODO 统计邻居节点，以及本节点其他属性数据情况。
                    #分析缺失情况
                    for step in range(tempdf.loc[row, 'steps']):  # 对于缺失数据较多区域进行遍历插值
                        # print(start_index + step)
                        insert_step = start_index + step  # 插值的位置
                        useXgboost = self._blockArea_missing_analyze(tempdf.loc[row, 'station_id'], insert_step,tempdf.loc[row, 'steps'],
                                                                     self.wind,intervals, type)
                        #使用xgboost数据填充
                        if useXgboost:
                            #获取xgboost 序号
                            index = self.originaldata.get_listIndex_by_id(tempdf.loc[row, 'station_id'])
                            # TODO 待测试
                            # 使用block_missing_filled_data数据填充，该数据在插值过程中
                            input_data = self.originaldata.list[index].iloc[insert_step,-6:].values.reshape(1,-1)
                            # estimate_data = self.xgboost_list[index].transform(input_data).reshape(-1) #type1 每个站点设置一个xgboost模型进行估计
                            # estimate_data = self.xgboostModel.transform(input_data).reshape(-1) #type2 全部站点设置一个xgboost模型进行估计
                            estimate_data = self.xgboostModel.transform_one_row(input_data).reshape(-1) # type3 全部站点设置一个xgboost模型，对于一行数据进行估计
                            for item in estimate_data:
                                if item < 0 or item>500:
                                    print('debug')
                            self.block_missing_filled_data.list[index].iloc[insert_step, -6:] = estimate_data
                            print('debug')
                        else: #不使用。 其他情况使用多视图插值方法， 多视图插值还失败，则丢弃
                            pass #其他情况在后续多视图插值中完成。

                        #TODO 使用idw
                        # # 修改数据到originData中
                        # data_matrix, target_index, dis_array = self.get_data_matrix(
                        #     tempdf.loc[row, 'station_id'], insert_step, intervals, type)
                        # # 获取idw,ses
                        # ses_d, _ = self.SES_Residual(data_matrix, target_index)  # 计算ses估计，单纯使用ses
                        # # ses_d_2, _ = self.SES(data_matrix, target_index)  # 计算ses估计，单纯使用ses
                        # idw_d, _ = self.IDW(data_matrix, dis_array, target_index)  # 计算IDW
                        # print('idw_d', idw_d)
                        # insert_data = np.nanmean([ses_d, idw_d])  # 使用idw和ses的好处是：防止ses方法对于block-missing data的插值趋于均值
                        # if np.isnan(insert_data):
                        #     print('debug')
                        # else:  # 将估计的数据填充回原始数据集中
                        #     # 获取station_id对应的id号
                        #     list_index = self.block_missing_filled_data.get_listIndex_by_id(
                        #         tempdf.loc[row, 'station_id'])
                        #     self.block_missing_filled_data.list[list_index].loc[insert_step, self.item] = insert_data
            df_ = self.block_missing_filled_data.get_mergedf() #将大片数据缺失，使用xgboost填充
            # block missing data output file
            block_missing_data_filled_filename = self.saveFilePath + 'block_missing_data_filled_file.csv'
            df_.to_csv(block_missing_data_filled_filename)
            #block missing填充完成，origin数据重新加载
            self.originaldata = df_clip(df_.copy(), 'station_id')
    def _estimate_by_mv_method(self,matrix):
        pass
    def multi_viewer_imputer(self,item = 'PM2.5',intervals='hour', type='middle'):
        #TODO 往self.originaldata里面写缺失值
        #对于某一属性进行插值
        #TODO 插值操作统一时刻优先
        #解决blockmissing 数据插值
        self._block_missing_op(intervals,type) #填充大范围数据缺失,使用
        #TODO block missing 插值后，self.originaldata数据为插值后数据
        #TODO 遍历全部时刻缺失点
        # TODO 1对时间缺失数据，进行排序
        tempdf = self.block_missing_log_time.copy().sort_values(by='index')
        tempdf = tempdf.reset_index(drop=True)  # 对于missing log数据按照原始数据的index排序，重新设置df的index
        # tempdf = tempdf[12549:]#debug
        for row in tempdf.index:
            print(row)
            station_id = tempdf.loc[row,'station_id']
            list_index = self.originaldata.get_listIndex_by_id(station_id)
            start_index = tempdf.loc[row, 'index']
            # step = tempdf.loc[row, 'steps']
            for step in range(tempdf.loc[row, 'steps']):
                # TODO 2遍历全部缺失数据行。
                #如果缺失数据大于
                if tempdf.loc[row, 'steps'] >2*self.wind:
                    break
                insert_index = start_index + step #具体index
                #TODO 判断是否有值，如果有，则是在block_missing节点采用xgboost插值
                if not np.isnan(self.originaldata.list[list_index].loc[insert_index, self.item]) \
                        and self.originaldata.list[list_index].loc[insert_index,self.item] >0: #TODO 添加对于xgboost插入负数值的判断，如果插入负数值则继续估计
                    #已经使用xgboost插值
                    pass
                else: #未插值成果

                    #TODO 判断未插值成功的原因是1:在大面积缺失数据中 或者 可采用multi-viewer插值
                    drop_missing_area = self._check_index_in_blocking_missing(station_id,insert_index,self.wind)  #TODO 查找当前index是否在block missing area
                    if drop_missing_area: #如果index处于mass lossing块中,大量数据缺失，直接舍弃
                        pass
                    else:
                        #TODO 使用多视图插值
                        esti_list, conf_list = self._get_four_estimation(station_id, insert_index) #返回四个估计列表
                        esti_list = np.array(esti_list).reshape(1,-1)
                        #使用线性回归模型，计算最终值
                        #y = sum(wx)+b

                        #TODO [当输入出现缺失值时，会报错], 直接加权计算，+ 偏置. 使用线性回归模型的话，。
                        # estimated_d = weighted_sum(esti_list,self.model_list[list_index].coef_)  + \
                        #                self.model_list[list_index].intercept_.reshape(-1)
                        if not np.isnan(esti_list).any():  #四个估计值都存在
                            estimated_d = self.model_list[list_index].predict(esti_list).reshape(-1)
                            print(estimated_d)
                            if estimated_d < 0:
                                estimated_d = np.max(esti_list)
                        else:  #TODO [当输入出现缺失值时，回报错] case：当前时间窗口内数据全部缺失;或者当前时刻其他节点数据都丢失;
                            print(esti_list)
                            #estimated_d = esti_list[0,0]
                            #TODO 使用非nan的数据
                            estimated_d = np.nanmean(esti_list)
                            if estimated_d < 0:
                                estimated_d = np.max(esti_list)

                        self.originaldata.list[list_index].loc[insert_index, self.item] = estimated_d
        df_ = self.originaldata.get_mergedf()  # 将大片数据缺失，使用xgboost填充
        # block missing data output file
        filled_filename = self.saveFilePath + 'filled_file.csv'
        df_.to_csv(filled_filename)

    def run(self):
        #TODO 每个站点数据训练回归模型
        self.train_LinearRegression_model()
        #TODO 对每个站点的训练xgboost树模型
        self.train_xgboost_model()
        # TODO 补全全部的缺失数据 xgboost + mul-viewer
        self.multi_viewer_imputer()
        # #sv-mvl: run方法两个模式: test对应插值方法比较，normal表示正常插值算法。
        # if type == 'test':
        #     #TODO 根据各个站点
        #     self.train_LinearRegression_model()  #每个站点数据训练回归模型
        #     #TODO补全全部的缺失数据
        #     self.multi_viewer_imputer()
        # else: #正常插值方法
        #     pass

# def st_viewer(oped_data, indexdf, item, outfile, neighbordf, neighbor_dis_df, knnVal=-1, wind=7):
    #oped_data:删除测试位置的数据集，indexdf 测试机的位置,
    #item: 插入值列名
    #outfile:输出数据名称， neighbordf：邻域表
    #neighbor_dis_df 领域距离表
    #knnVal：K近邻的数量，取-1时表示选择全部邻居
    #wind:时间窗口的大小
    #check block loss
    # return 0,0
if __name__ == '__main__':
    #1 compareomputingMethods.py check_loss
    #2 delete test_index data
    #3 insert method.
    #4 compute result

    check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv",
                                index_col='Unnamed: 0')
    neighbor_info = neighbor_info.sort_index()  #重新排序index
    neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv",
                                    index_col='Unnamed: 0')
    item = [ 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2' ]
    st_d = st_mvl(oped_data, check_list, 'PM2.5', 'auxiliaryFile/1718',neighbor_info,
                  neighbor_dis_info,knnVal=7,wind=11)
    st_d.run()

    #mae, mre = st_viewer(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv',neighbor_info, neighbor_dis_info,knnVal=-1,wind=7)
    #mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv',neighbor_info, neighbor_dis_info,knnVal=-1,wind=7)
    #print(mae, mre)