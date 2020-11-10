from pathlib import Path
from sklearn import linear_model
# from ModelBasedImputer.MissingImputer import MissingImputer  #封装了xgboost等多种模型
# import xgboost as xgb
# from hyperopt import hp
# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from ImputerOp.multiobjXGBoost import MultiXGboost
import warnings
from utils.func import getMAE, getMRE, _diff
from utils.multi_view_impute_fn import *
from utils.Structure import df_clip
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

class IMA():
    def __init__(self,oped_data, indexdf, item, subitem, outfilepath, neighbordf, neighbor_dis_df, knnVal=-1, wind=7,testid_size = -1):
        #TODO item
        self.originaldata = df_clip(oped_data, 'station_id')  #数据按照station_id划分 df_clip类
        self.test_index = indexdf
        self.item = item  #使用参数列表 [ 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2' ]
        self.columns = []  #获取item对应的列号
        for item_ in self.item:
            for i in range(self.originaldata.list[0].columns.size):  # 获取item对应的列号
                if self.originaldata.list[0].columns.values[i] == item_:
                    self.columns.append(i)
                    break
        self.testItem = subitem  #记录测试实验对象，真实插值过程，其值等于self.item
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
        # self.weight = np.ones([self.neighborInfo.shape[0],4]) * 0.25 #待确定
        # self.bias = np.random.random([self.neighborInfo.shape[0]])

        #self.LRtraining_data = [pd.DataFrame([],columns=['idw','ses','ucf','icf','y']) for i in range(self.test_station_size)] #存储训练数据
        self.LRtraining_data = [[pd.DataFrame([],columns=['idw','ses','ucf','icf','y']) for i in range(self.test_station_size)]
                                for j in range(len(self.item))]
        self.model_list = [[linear_model.LinearRegression(n_jobs=10) for i in range(self.test_station_size)]
                           for j in range(len(self.item))]

        # self.xgboost_list = [MissingImputer(max_iter = 100, ini_fill= True, model_reg='xgboost', model_clf='xgboost')
        #                 for i in range(self.test_station_size)]  #为每个检测站点训练一个xgboost模型  #missinImputer method

        params = {'n_estimators': 360, 'colsample_bytree': 0.7166589051315236, 'gamma': 0.012954988155888664,
                  'learning_rate': 0.09067065786724686, 'max_depth': 4, 'reg_alpha': 0.35325351285730894,
                  'subsample': 0.8719641998030547}

        # self.xgboost_list = [MultiXGboost(params) for i in range(self.test_station_size)]  # 为每个检测站点训练一个xgboost模型 为每个站点设置一个模型
        # self.xgboostModel = MissingImputer(max_iter = 30, ini_fill= True, model_reg='xgboost', model_clf='xgboost')
        self.xgboostModel = MultiXGboost(params)  #使用


        # self.xgboost_filled_df = df_clip(oped_data.copy(), 'station_id') #记录xgboost填充的数据
        #self.learningRate = 1e-5
        #TODO 用于存储缺失数据情况,列表表示存储不同的属性
        self.block_missing_log_time = [pd.DataFrame([],columns=['station_id','index','steps']) for _ in range(len(self.item))] #统计每个站点上连续时间缺失的情况
        self.block_missing_log_space = [pd.DataFrame([], columns=['index']) for _ in range(len(self.item))] #统计同一时刻，全体节点数据缺失的情况
        self.space_loss_time = [pd.DataFrame([],columns=['index','steps']) for _ in range(len(self.item))]  #统计全体数据缺失的时刻，以及连续步长
        self.check_blockmissing_byItem() #获取上面三个变量的值 TODO 由于xgboost插值过程会
        # self.check_file() #核对中间文件是否已经存储，如果存储，则加载，否则再相应程序中计算
        self.block_missing_filled_data = df_clip(oped_data.copy(), 'station_id')   #存储添加block missing的数据

   #TODO 获取数据集中某个属性的数据
    def construct_data_matrix_by_columnsname(self, param):
        #TODO
        # 按照item name重构data matrix
        # 返回matrix矩阵中对应位置是否为nan值
        result_df = pd.DataFrame()
        first = True
        for data_byid in self.originaldata.list:
            if first:
                result_df = data_byid[param]
                first = False
            else:
                result_df = pd.concat([result_df, data_byid[param]],axis=1)
        # print(result_df)
        result_df.columns = self.originaldata.id
        return result_df.isna()

    # TODO 获取原始数据缺失情况
    def _check_blockmissing(self,param,param_index):
        #TODO
        # item_name表示核对缺失值的属性，index_表示统计属性存储的list索引
        #重构 数据矩阵
        nan_matrix = self.construct_data_matrix_by_columnsname(param)
        #TODO 1 对nan_matrix 做行统计，做列统计。
        # 大于window数量的缺失为时间block缺失
        block_missing_file_time = self.saveFilePath + param + 'block_missing_log_time.csv'
        block_missing_file_time_path = Path(block_missing_file_time)
        file1_exsit = False

        block_missing_file_space = self.saveFilePath + param + 'block_missing_log_space.csv'
        block_missing_file_space_path = Path(block_missing_file_space)
        file2_exsit = False

        space_loss_time = self.saveFilePath + param + 'space_missing_time.csv'
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
                            self.block_missing_log_time[param_index] = self.block_missing_log_time[param_index].append(series, ignore_index=True)
                            first_nan = True
                            steps = 0
                        else: #TODO 2.2处于无缺失区域
                            # 处于无缺失区域
                            pass
                if steps >0: #TODO 添加末尾的数据
                    series = pd.Series([id, start, steps], index=['station_id', 'index', 'steps'])
                    self.block_missing_log_time[param_index] = self.block_missing_log_time[param_index].append(series, ignore_index=True)
            self.block_missing_log_time[param_index].to_csv(block_missing_file_time)
                    #check loss
        else:
            self.block_missing_log_time[param_index] = pd.read_csv(block_missing_file_time,index_col='Unnamed: 0')
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
            self.block_missing_log_space[param_index]['index'] = index
            self.block_missing_log_space[param_index].to_csv(block_missing_file_space)
            #check row empty
        else:
            self.block_missing_log_space[param_index] = pd.read_csv(block_missing_file_space, index_col='Unnamed: 0')
        # print(self.block_missing_log_space[index_].head())
        #TODO 3 统计空间块缺失的时间连续性
        if not file3_exsit:

            temp_df = self.block_missing_log_space[param_index]
            temp_df['diff'] = self.block_missing_log_space[param_index].diff() # diff=1表示连续
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
                        self.space_loss_time[param_index] = self.space_loss_time[param_index].append(series, ignore_index=True)
                        start = temp_df.loc[row,'index']
                        steps = 1

                    else:
                        steps += 1

            #保存最后一条数据
            series = pd.Series([start, steps], index=['index', 'steps'])
            self.space_loss_time[param_index] = self.space_loss_time[param_index].append(series, ignore_index=True)
            self.space_loss_time[param_index].to_csv(space_loss_time)
        else: #文件已经存在，直接加载
            self.space_loss_time[param_index] = pd.read_csv(space_loss_time, index_col='Unnamed: 0')

        print('loading loss log file end')

    def check_blockmissing_byItem(self):
        # TODO 核对数据集中各属性缺失情况
        for param_index, param in enumerate(self.item,start=0):
            print('Check blockmissing for',param)
            self._check_blockmissing(param, param_index)
    def _get_data_matrix_index(self, query_row, rowSize, intervals, type):
        #TODO 获取插值矩阵的index
        # query_row:缺失值所在行 rowSize: 总行数 intervals:时间检测 type: middle, end
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
        # matrix_begin_index = 0
        # matrix_end_index = 0
        # index = []
        #TODO method1
        if type =='middle':
            index = [query_row + (i-int(self.wind/2))* intervals_ for i in range(self.wind)]
            target_index = int(self.wind/2)
        else:
            index = [query_row - (self.wind-1 -i) * intervals_ for i in range(self.wind)]
            target_index = self.wind
        # print(index)
        #TODO delete < 0 的index
        filtered_list = [i for i in index if i>=0 and i < rowSize] #TODO 调整索引列表，将越界的索引去除
        target_index = target_index - (self.wind - len(filtered_list))  #TODO 目标索引位置改变
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
    def get_data_matrix(self, param, station_id, query_row, intervals='hour',type = 'middle'):
        #TODO 返回插值矩阵的index
        #获取item 对于的index
        columns = 0
        for index_, item_ in enumerate(self.item, start=0): #todo 根据itemname 获取index_
            if item_ == param:
                columns = self.columns[index_]
                break

        index,target_index = self._get_data_matrix_index(query_row, self.datalength, intervals, type) #获取数据列表，以及数据所在行
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
            if list_index ==-1:
                #list_index =-1表示该ID不在数据集中
                continue
            series = self.originaldata.list[list_index].iloc[index, columns:columns + 1].copy()
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
        dis = dis_all[neighborlist] #获取dis
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
            #TODO 1016
            for i in range(len(diff)):
                if i > target_index:
                    diff[i]*=-1
            #TODO 1016 大于目标区域的趋势为数值
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

    def _get_four_estimation(self, param, station_id, row, intervals='hour', type='middle'):
        # TODO 获取四个估计值
        #TODO 使用四种估计模型，估计某一时刻
        # 输入参数：
        data_matrix, target_index, neighbor_dis = self.get_data_matrix(param, station_id, row, intervals=intervals, type=type)
        # TODO 四种估计方法，
        # 返回四种方法的估计值数组，以及置信值数组
        # print(data_matrix)
        # print(neighbor_dis)
        idw_d, idw_conf = self.IDW(data_matrix, neighbor_dis, target_index)
        # ses_d, ses_conf = self.SES(data_matrix, target_index) #TODO 无残差
        # ses_d, ses_conf = self.SES_Residual(data_matrix, target_index) #TODO 1016 不采用残差
        ses_d, ses_conf = self.SES(data_matrix, target_index)
        ucf_d, ucf_conf = self.UCF(data_matrix, target_index)
        icf_d, icf_conf = self.ICF(data_matrix, target_index)

        return [idw_d,ses_d,ucf_d,icf_d], [idw_conf,ses_conf,ucf_conf,icf_conf]  #TODO 置信值备用


    def get_LinearR_trainingdata(self, param_index, param, station_id,intervals='hour', type='middle'):
        # TODO 获取LR模型训练数据
        #TODO 该函数实现对于某一站点的数据，进行无缺失部分的提取，线性回归模型的训练
        # 原始数据存储位置 self.originaldata.list[i]，处理后数据存储位置
        # self.LRtraining_data[i]中
        listIndex = self.originaldata.get_listIndex_by_id(station_id)
        if self.LRtraining_data[param_index][listIndex].shape[0] >0:  #判断数据是否已经存在
            return self.LRtraining_data[param_index][listIndex]
        else:
            outfile_name = self.saveFilePath + str(station_id) + param + 'trainLRdata.csv'
            outfile_name_path = Path(outfile_name)
            #判断数据文件是否存在
            if outfile_name_path.exists(): #如果文件存在则加载该文件，减少计算时间
                self.LRtraining_data[param_index][listIndex] = pd.read_csv(outfile_name_path,index_col='Unnamed: 0')
                print(station_id,'loading training data')

            else: #否则计算存储文件
                #计算数据
                # 获取存储键值
                for row in self.originaldata.list[listIndex].index: #
                    #求取训练集：
                    esti_list, conf_list = self._get_four_estimation(param, station_id, row)
                    esti_list.append(self.originaldata.list[listIndex].loc[row, param])  #添加真实值
                    series = pd.Series(esti_list, index=['idw','ses','ucf','icf','y'])
                    self.LRtraining_data[param_index][listIndex] = \
                        self.LRtraining_data[param_index][listIndex].append(series, ignore_index=True)
                #拼接数据
                # outfile_name = self.saveFilePath + station_id +'trainLRdata.csv'
                self.LRtraining_data[param_index][listIndex].to_csv(outfile_name)


    def train_LinearRegression_model(self, param_index, param):
        #TODO 使用sklearn计算回归模型
        # 通过已有数据集训练加权模型的权重，偏置
        # 添加判断参数文件是否存在判断，如果存在则加载。
        # 否则重新训练
        for i in range(self.test_station_size):    #遍历station站点
            # station_df = self.originaldata.list[i]  # 获取数据
            station_id = self.originaldata.id[i]  # 获取id
            model_parms_ = self.saveFilePath + str(station_id) + param + '_LR_.pkl'
            model_parms_path = Path(model_parms_)
            if model_parms_path.exists():   #
                print(station_id, param,'LR  params file exist')
                # 加载模型 ＃prams = joblib.load("prams.pkl")
                self.model_list[param_index][i] = joblib.load(model_parms_path)
            else: #
                print(station_id,param, 'LR params not exist')
                #模型文件不存在
                #TODO 模型 self.model_list[0]
                # 提取训练数据
                self.get_LinearR_trainingdata(param_index, param, station_id) #
                #训练模型
                #获取训练数据，删除有nan值的行
                sub_df = self.LRtraining_data[param_index][i].copy()
                sub_df = sub_df.dropna()
                #分割输入数据，和标签数据
                input_d = sub_df[['idw','ses','ucf','icf']].values
                y_d = sub_df[['y']].values
                #训练模型
                self.model_list[param_index][i].fit(input_d,y_d)
                ## 保存模型 joblib.dump(prams,'prams.pkl')
                joblib.dump(self.model_list[param_index][i], model_parms_path)
                # print(self.model_list[i].coef_)


    def train_xgboost_model(self):
        #TODO 存在问题，估计值存在负数
        #TODO 两种策略：1:每个站点训练一个模型  1：全部数据训练一个模型
        # 整理训练集，训练模型 self.xgboostModel。
        model_params = self.saveFilePath+ "xgb.model.dat"
        model_params_path = Path(model_params)
        if model_params_path.exists():
            # TODO 模型存在，则加载模型
            print('XGboost model exists，loading model...')
            self.xgboostModel = joblib.load(model_params_path)
            # all_train_df = self.originaldata.get_mergedf().copy()
            # filled_df = self.xgboostModel.transform(all_train_df.iloc[:, -len(self.item):])
            # all_train_df.iloc[:, -len(self.item):] = filled_df
            # all_train_df.to_csv(self.saveFilePath + '1.csv')
        else:
            # TODO 否则则训练模型
            print('XGboost model not exists, training model...')
            all_train_df = self.originaldata.get_mergedf().copy() #将按列表划分的dataframe合并
            train_df = all_train_df.dropna(axis=0,how='any').reset_index(drop=True) #丢弃含有缺失项的数据
            # print(train_df.iloc[:,-len(self.item):])
            # self.xgboostModel.fit(train_df.iloc[:,-len(self.item):])  #
            #插值数据在最后一行
            self.xgboostModel.fit(all_train_df.iloc[:,-len(self.item):])  #TODO 两种方案，使用非空的数据训练xgboost, 2使用全部数据训练

            joblib.dump(self.xgboostModel,self.saveFilePath+"xgb.model.dat")
            # test
            # xgb = joblib.load(model_params)
            # filled_df = xgb.transform(all_train_df.iloc[:,-len(self.item):])
            # all_train_df.iloc[:,-len(self.item):] = filled_df
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
        #         # print(sub_df.iloc[6, -len(self.item):].values.reshape(1, -1))
        #         # input = sub_df.iloc[6, -len(self.item):].values.reshape(1, -1)
        #         # estimate = self.xgboost_list[i].transform(input)
        #     else:
        #         print(station_id,'prams params not exist')
        #         sub_df = self.originaldata.list[i].copy()
        #         numerialMatric = sub_df.iloc[:, -len(self.item):]  # 获取删除测试数据后的数据区域
        #         self.xgboost_list[i].fit(numerialMatric)
        #         self.xgboost_filled_df.list[i].iloc[:,-len(self.item):] = self.xgboost_list[i].transform(numerialMatric)
        #         self.xgboost_filled_df.list[i].to_csv(self.saveFilePath+str(i)+'xgboost_filled.csv')
        #         joblib.dump(self.xgboost_list[i],model_parms_path)
        #         #TODO 使用xgboost，预测一个值
        #         # estimate = self.xgboost_list[i].transform(sub_df.iloc[0:6, -len(self.item):] )
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
    def  _blockArea_missing_analyze(self,station_id, param_index, row, step, w, intervals='hour', type='middle'):
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
        # 使用情况：1：同时刻其他站点该属性完全缺失 2：属性邻近发生连续缺失(1008修改为时间域单一缺失便插值)
        # if row in self.block_missing_log_space[param_index].values or step > w: #TODO 判断row是否在全部站点缺失数据中，且所处的缺失的时间窗口大于w


        if row in self.block_missing_log_space[param_index].values and step > w: #TODO 1008判断row是否在全部站点缺失数据中,并且时间窗口内数据缺失一半多
            #该时间点，全部节点数据缺失
            #TODO check 本地时刻其他属性数据是否全部缺失
            id_index = self.originaldata.get_listIndex_by_id(station_id) #list_index
            # print(self.originaldata.list[id_index].loc[row,['PM2.5','PM10','NO2','CO','O3','SO2']])

            loss_all = self.originaldata.list[id_index].loc[row,self.item].isna().all() #获取该时间，节点数据
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
    def _get_param_index_in_paramList(self,parm):
        assert parm in self.item, "Param is not in Params list"
        for i,item_ in enumerate(self.item,start=0):
            if item_ == parm:
                return i

    def _check_index_in_blocking_missing(self,params_index, check_row, w):
        #TODO 该函数用来检测某一行，是否存在block missing区域
        # 返回： 是否为block missing 区域数据：flag
        search_df = self.space_loss_time[params_index].copy()
        search_df = search_df.sort_values(by='index')
        flag = False
        for row in search_df.index:
            start_index = search_df.loc[row,'index']
            end_index = search_df.loc[row,'index'] + search_df.loc[row,'steps'] -1
            if check_row >= start_index and check_row <=end_index:
                #if search_df.loc[row,'steps'] >w:  #TODO 1015 w 如果缺失数据过多，则返回丢弃标志
                flag = True  #TODO 1015 处于块缺失中，则多视图插值只有ses有效
                break
            if check_row < start_index:
                break
        return flag
    #TODO 数据插值操作
    def block_missing_op(self, intervals='hour', type='middle'):
        #TODO 对整体数据集做xgboost插值处理
        block_missing_data_filled_filename = self.saveFilePath + 'block_missing_data_filled_file.csv'
        block_missing_data_filled_filename_path = Path(block_missing_data_filled_filename)
        #TODO 为加快调试速度，处理文件将存储在本地，可直接加载
        if block_missing_data_filled_filename_path.exists():  # debug 模式，为避免重复计算，暂存计算中间量。如果block缺失数据已经存储，直接加载，当参数修改后，文件删除，重新计算
            block_missing_dara_filled_df = pd.read_csv(block_missing_data_filled_filename_path, index_col='Unnamed: 0')
            self.block_missing_filled_data = df_clip(block_missing_dara_filled_df, 'station_id')
            self.originaldata = df_clip(block_missing_dara_filled_df.copy(), 'station_id')
        else:

            #TODO 分别对不同属性的进行块缺失xgboost插值，对于block_missing_log_time 按照进行index排序
            for param_index, param in enumerate(self.item, start =0):
                tempdf = self.block_missing_log_time[param_index].copy().sort_values(by='index')  #TODO 此处使用 PM2.5的列为主
                tempdf = tempdf.reset_index(drop=True)  # 对于missing log数据按照原始数据的index排序，重新设置df的index
                # print(tempdf.head())
                for row in tempdf.index:  #TODO 时间维度上，遍历全部缺失的数据
                    if(row%3000==0):
                        print('Block Missing Running process: ',param,' ',row,'/',len(tempdf))
                    # if tempdf.loc[row, 'steps'] >= self.wind:  # 对于时间上，大于缺失插值窗口大小的数据进行预先填充, 改为对于时许缺失数据全部全部分析
                    #TODO 对于数据进行插值
                    if tempdf.loc[row, 'steps'] >= 1:  # 全部有效
                        # print(tempdf.loc[row,'index'])
                        # 对于连续的数据缺失填充 idw ses
                        start_index = tempdf.loc[row, 'index'] #获得原始数据所在行
                        #TODO 统计邻居节点，以及本节点其他属性数据情况。
                        #分析缺失情况
                        for step in range(tempdf.loc[row, 'steps']):  # 对于缺失数据较多区域进行遍历插值
                            # print(start_index + step)
                            insert_step = start_index + step  # 插值的位置
                            #TODO 使用XGBoost的方法  if row in self.block_missing_log_space[param_index].values and step > w/2:
                            useXgboost = self._blockArea_missing_analyze(tempdf.loc[row, 'station_id'], param_index, insert_step, tempdf.loc[row, 'steps'],
                                                                         self.wind, intervals, type) #分析是否使用xgboost
                            #使用xgboost数据填充
                            if useXgboost:
                                #获取xgboost 序号
                                index = self.originaldata.get_listIndex_by_id(tempdf.loc[row, 'station_id'])
                                # TODO 待测试
                                # 使用block_missing_filled_data数据填充，该数据在插值过程中
                                input_data = self.originaldata.list[index].iloc[insert_step,-len(self.item):].values.reshape(1,-1)
                                # estimate_data = self.xgboost_list[index].transform(input_data).reshape(-1) #type1 每个站点设置一个xgboost模型进行估计
                                # estimate_data = self.xgboostModel.transform(input_data).reshape(-1) #type2 全部站点设置一个xgboost模型进行估计
                                estimate_data = self.xgboostModel.transform_one_row(input_data).reshape(-1) #TODO type3 全部站点设置一个xgboost模型，对于一行数据进行估计
                                # for item_ in estimate_data:
                                    # if item_ < 0 or item_>500:
                                    #     print('debug')
                                self.block_missing_filled_data.list[index].iloc[insert_step, -len(self.item):] = estimate_data
                                # print('debug')
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
            self.originaldata = df_clip(df_.copy(), 'station_id') #将填充后的数据覆盖到originaldata中
    # def _estimate_by_mv_method(self,matrix):
    #     pass
    def multi_viewer_imputer_bk1015(self, param, column_index, param_index,intervals='hour', type='middle'):
        #TODO params:表示属性名称 param_index： 表示数量的索引，适用于对应训练模型的列表索引
        # 插值后self.originaldata里面写缺失值
        filled_filename = self.saveFilePath + param + 'filled_file.csv'
        filled_filename_path = Path(filled_filename)
        if filled_filename_path.exists(): #TODO 如果某属性插值文件存在
            #TODO 加载filename 文件到self.originaldata中
            df_ = pd.read_csv(filled_filename_path, index_col='Unnamed: 0')
            self.originaldata = df_clip(df_,'station_id')
        else: #文件不存在

            # 遍历全部时刻缺失点
            # 1对时间缺失数据，进行排序
            tempdf = self.block_missing_log_time[param_index].copy().sort_values(by='index')
            tempdf = tempdf.reset_index(drop=True)  # 对于missing log数据按照原始数据的index排序，重新设置df的index
            # tempdf = tempdf[12549:]#debug
            for row in tempdf.index:
                if row %1000 ==0:
                    print('Process: ',row,'/',len(tempdf))
                station_id = tempdf.loc[row,'station_id']
                list_index = self.originaldata.get_listIndex_by_id(station_id)
                start_index = tempdf.loc[row, 'index']
                # step = tempdf.loc[row, 'steps']
                for step in range(tempdf.loc[row, 'steps']):
                    # TODO 2遍历全部缺失数据行。
                    #如果缺失数据大于2w,跳出本层循环
                    if tempdf.loc[row, 'steps'] > 2 * self.wind : #时间维度大范围数据缺失
                        #TODO 1014 xgboost
                        break
                    else:
                        insert_index = start_index + step #具体index
                        #TODO 判断是否有值，如果有，则是在block_missing节点采用xgboost插值
                        if not np.isnan(self.originaldata.list[list_index].loc[insert_index, param]) \
                                and self.originaldata.list[list_index].loc[insert_index,param] >0: #TODO 添加对于xgboost插入负数值的判断，如果插入负数值则继续估计
                            #已经使用xgboost插值
                            pass
                        else: #未插值成果
                            #TODO 判断未插值成功的原因是1:在大面积缺失数据中 或者 可采用multi-viewer插值
                            drop_missing_area = self._check_index_in_blocking_missing(param_index, insert_index,self.wind)  #TODO 查找当前index是否在block missing area
                            if drop_missing_area: #如果index处于mass lossing块中,大量数据缺失，直接舍弃
                                pass
                            else:
                                #TODO 使用多视图插值 idw_d,ses_d,ucf_d,icf_d
                                esti_list, conf_list = self._get_four_estimation(param, station_id, insert_index) #返回四个估计列表
                                esti_list = np.array(esti_list).reshape(1,-1)
                                #使用线性回归模型，计算最终值
                                #y = sum(wx)+b
                                #TODO [当输入出现缺失值时，会报错], 直接加权计算，+ 偏置. 使用线性回归模型的话，。
                                # estimated_d = weighted_sum(esti_list,self.model_list[list_index].coef_)  + \
                                #                self.model_list[list_index].intercept_.reshape(-1)
                                if not np.isnan(esti_list).any():  #四个估计值都存在
                                    estimated_d = self.model_list[param_index][list_index].predict(esti_list).reshape(-1)
                                    # print(estimated_d)
                                    if estimated_d < 0:
                                        estimated_d = np.max(esti_list)
                                else:  #TODO [当输入出现缺失值时，回报错] case：当前时间窗口内数据全部缺失;或者当前时刻其他节点数据都丢失;
                                    # print(esti_list)
                                    #estimated_d = esti_list[0,0]
                                    #TODO 使用非nan的数据
                                    #使用xgboost辅助估计
                                    estimated_d = np.nanmean(esti_list)
                                    if estimated_d < 0:
                                        estimated_d = np.max(esti_list)

                                self.originaldata.list[list_index].loc[insert_index, param] = estimated_d
            df_ = self.originaldata.get_mergedf()  # 将大片数据缺失，使用xgboost填充
            # block missing data output file
            df_.to_csv(filled_filename)

    def multi_viewer_imputer(self, param, column_index, param_index, intervals='hour', type='middle'):
        # TODO params:表示属性名称 param_index： 表示数量的索引，适用于对应训练模型的列表索引
        # 插值后self.originaldata里面写缺失值
        filled_filename = self.saveFilePath + param + 'filled_file.csv'
        filled_filename_path = Path(filled_filename)
        if filled_filename_path.exists():  # TODO 如果某属性插值文件存在
            # TODO 加载filename 文件到self.originaldata中
            df_ = pd.read_csv(filled_filename_path, index_col='Unnamed: 0')
            self.originaldata = df_clip(df_, 'station_id')
        else:  #TODO 插值文件不存在
            # 遍历全部时刻缺失点
            # 1对时间缺失数据，进行排序
            tempdf = self.block_missing_log_time[param_index].copy().sort_values(by='index')
            tempdf = tempdf.reset_index(drop=True)  # 对于missing log数据按照原始数据的index排序，重新设置df的index
            # tempdf = tempdf[12549:]#debug
            for row in tempdf.index:
                if row % 3000 == 0:
                    print('Process: ', row, '/', len(tempdf))
                station_id = tempdf.loc[row, 'station_id']
                list_index = self.originaldata.get_listIndex_by_id(station_id)
                start_index = tempdf.loc[row, 'index']
                # step = tempdf.loc[row, 'steps']
                for step in range(tempdf.loc[row, 'steps']):
                    # TODO 2遍历全部缺失数据行。
                    # 如果缺失数据大于2w,跳出本层循环
                    # if tempdf.loc[row, 'steps'] > self.wind/2:  #TODO 1015 时间维度大范围数据缺失
                    #     # TODO 1014 xgboost
                    #     break
                    # else:
                    #     insert_index = start_index + step  # 具体index
                    #     # TODO 判断是否有值，如果有，则是在block_missing节点采用xgboost插值
                    #     if not np.isnan(self.originaldata.list[list_index].loc[insert_index, param]) \
                    #             and self.originaldata.list[list_index].loc[
                    #         insert_index, param] > 0:  # TODO 添加对于xgboost插入负数值的判断，如果插入负数值则继续估计
                    #         # 已经使用xgboost插值
                    #         pass
                    #     else:  # 未插值成果
                    #         # TODO 判断未插值成功的原因是1:在大面积缺失数据中 或者 可采用multi-viewer插值
                    #         drop_missing_area = self._check_index_in_blocking_missing(param_index, insert_index,
                    #                                                                   self.wind)  # TODO 查找当前index是否在block missing area
                    #         if drop_missing_area:  # 如果index处于mass lossing块中,大量数据缺失，直接舍弃
                    #             pass
                    #         else:
                    #             # TODO 使用多视图插值 idw_d,ses_d,ucf_d,icf_d
                    #             esti_list, conf_list = self._get_four_estimation(param, station_id,
                    #                                                              insert_index)  # 返回四个估计列表
                    #             esti_list = np.array(esti_list).reshape(1, -1)
                    #             # 使用线性回归模型，计算最终值
                    #             # y = sum(wx)+b
                    #             # TODO [当输入出现缺失值时，会报错], 直接加权计算，+ 偏置. 使用线性回归模型的话，。
                    #             # estimated_d = weighted_sum(esti_list,self.model_list[list_index].coef_)  + \
                    #             #                self.model_list[list_index].intercept_.reshape(-1)
                    #             if not np.isnan(esti_list).any():  # 四个估计值都存在
                    #                 estimated_d = self.model_list[param_index][list_index].predict(esti_list).reshape(
                    #                     -1)
                    #                 # print(estimated_d)
                    #                 if estimated_d < 0:
                    #                     estimated_d = np.max(esti_list)
                    #             else:  # TODO [当输入出现缺失值时，回报错] case：当前时间窗口内数据全部缺失;或者当前时刻其他节点数据都丢失;
                    #                 # print(esti_list)
                    #                 # estimated_d = esti_list[0,0]
                    #                 # TODO 使用非nan的数据
                    #                 # 使用xgboost辅助估计
                    #                 estimated_d = np.nanmean(esti_list)
                    #                 if estimated_d < 0:
                    #                     estimated_d = np.max(esti_list)
                    #
                    #             self.originaldata.list[list_index].loc[insert_index, param] = estimated_d
                    #判断数据是否已经插值成功
                    insert_index = start_index + step
                    if not np.isnan(self.originaldata.list[list_index].loc[insert_index, param]):
                        #TODO 数据已经被xgboost预插值
                        pass
                    else:
                        #TODO 数据未被插值
                        in_block_missing_area = self._check_index_in_blocking_missing(param_index, insert_index,self.wind)
                        in_time_series_missing_area = True if tempdf.loc[row, 'steps'] > self.wind else False
                        #当前站点，属性采集的未缺失情况
                        series = self.originaldata.list[list_index].iloc[insert_index,-len(self.item):].values
                        series_not_nan_rate = 1 - get_nan_rate(series)
                        # TODO 查找当前index是否在block missing area
                        # TODO case1 空间缺失，时间域连续缺失
                        if in_block_missing_area and in_time_series_missing_area:
                            #使用xgboost预先插值
                            # print('Mass Block Missing')
                            pass
                        # TODO case2 空间缺失，时间域不连续缺失
                        if in_block_missing_area and not in_time_series_missing_area:
                            #优先ses插值，如果缺失数据过多则使用xgboost
                            data_matrix, target_index, neighbor_dis = self.get_data_matrix(param, station_id, insert_index)
                            ses_d, ses_conf = self.SES_Residual(data_matrix, target_index) #TODO 1016
                            # ses_d, ses_conf = self.SES(data_matrix, target_index)
                            estimated_d = ses_d
                            #使用xgboost估计
                            # TODO 比较两个方法的可性度：
                            if series_not_nan_rate > ses_conf:
                                estimated_list = self.xgboostModel.transform_one_row(series.reshape(1,-1)).reshape(
                                    -1)  # TODO type3 全部站点设置一个xgboost模型，对于一行数据进行估计
                                index_ = self._get_param_index_in_paramList(param)
                                estimated_d = estimated_list[index_]
                            self.originaldata.list[list_index].loc[insert_index, param] = estimated_d
                        # TODO case3 空间不缺失，时间域连续缺失
                        if not in_block_missing_area and in_time_series_missing_area:
                            data_matrix, target_index, neighbor_dis = self.get_data_matrix(param, station_id,
                                                                                           insert_index)
                            idw_d, idw_conf = self.IDW(data_matrix, neighbor_dis, target_index)
                            estimated_d = idw_d
                            #使用xgboost
                            # TODO 比较两个方法的可性度：
                            if series_not_nan_rate > idw_conf:
                                estimated_list = self.xgboostModel.transform_one_row(series.reshape(1,-1)).reshape(
                                    -1)  # TODO type3 全部站点设置一个xgboost模型，对于一行数据进行估计
                                index_ = self._get_param_index_in_paramList(param)
                                estimated_d = estimated_list[index_]
                            self.originaldata.list[list_index].loc[insert_index, param] = estimated_d
                        # TODO case4 空间不缺失，时间域不连续缺失
                        if not in_block_missing_area and not in_time_series_missing_area:
                            esti_list, conf_list = self._get_four_estimation(param, station_id,
                                                                             insert_index)  # 返回四个估计列表
                            esti_list = np.array(esti_list).reshape(1, -1)
                            if not np.isnan(esti_list).any():  # 四个估计值都存在
                                estimated_d = self.model_list[param_index][list_index].predict(esti_list).reshape(-1)
                                # print(estimated_d)
                                if estimated_d < 0:
                                    estimated_d = np.max(esti_list)

                            else:  # TODO [当输入出现缺失值时，回报错] case：当前时间窗口内数据全部缺失;或者当前时刻其他节点数据都丢失;
                                # print(esti_list)
                                # estimated_d = esti_list[0,0]
                                # TODO 使用非nan的数据
                                # 使用xgboost辅助估计
                                estimated_d = np.nanmean(esti_list)
                                if estimated_d < 0:
                                    estimated_d = np.max(esti_list)
                            self.originaldata.list[list_index].loc[insert_index, param] = estimated_d
            df_ = self.originaldata.get_mergedf()  # 将大片数据缺失，使用xgboost填充
            # block missing data output file
            df_.to_csv(filled_filename)

    def final_check(self):
        #TODO 此函数为最终插值检测遗漏部分，以及异常高，及负数部分检测。简单指数平滑
        for list_index, subdf_ in enumerate(self.originaldata.list,start=0):
            for row in subdf_.index: #遍历每一行
                loss_ = subdf_.loc[row, self.item].isna()
                if loss_.any(): #如果还有缺失值
                    # print(subdf_.loc[row,self.item])
                    for column, flag in enumerate(loss_, start=0):
                        if flag: #缺失
                            # print(subdf_.loc[row-2:row+2,self.item[column]])
                            series = subdf_.loc[row-5:row+5,self.item[column]]
                            nan_num = series.isna().sum() # 该列nan值的数量
                            nan_rate = nan_num / len(series)  # nan值的比例
                            if nan_rate>1/2:
                                if loss_.all():
                                    pass
                                else:

                                    input_data = subdf_.loc[row, self.item].values.reshape(1,-1)
                                    estimate_data = self.xgboostModel.transform_one_row(input_data).reshape(-1)  # TODO type3 全部站点设置一个xgboost模型，对于一行数据进行估计
                                    self.originaldata.list[list_index].loc[row,-len(self.item):] = estimate_data
                            else:
                                esti_d,_ = self.SES(series.values.reshape(-1,1), int(len(series)/2))
                                self.originaldata.list[list_index].loc[row,self.item[column]] =  esti_d
                                #print(int(len(series)/2))
                            # imputed_d = self.SES(data, target_index)
                        else:
                            pass
                    #返回缺失值的位置
                else:
                    pass

                #检测插入值是否为负数
                for column, val in enumerate(subdf_.loc[row, self.item],start = 0):
                    if not loss_[column] and  val < 0: #数据非空，并且小于0
                        series = subdf_.loc[row-2:row+2,self.item[column]]
                        self.originaldata.list[list_index].loc[row, self.item[column]] = np.nanmean(series)
                        # print(np.nanmean(series))



    def compute_score(self,item):
        #TODO 本函数用于计算目标插入值的偏差值
        # 比较 self.test_index 对应值与self.originaldata中对应数值之间的mse,
        for row in self.test_index.index:
            if row%5000==0:
                print('Process')
            list_index = self.originaldata.get_listIndex_by_id(self.test_index.loc[row,'station_id'])
            check_row = self.test_index.loc[row,'row']
            self.test_index.loc[row,'esti_data'] = self.originaldata.list[list_index].loc[check_row,item]
        reald = self.test_index['data'].values
        estid = self.test_index['esti_data'].values
        self.test_index.to_csv(self.saveFilePath+'test_score.csv')
        mae = getMAE(reald,estid)
        mre = getMRE(reald,estid)
        with open('result/output_result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
            # print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
            print('IMA:','MAE', mae, 'MRE', mre, file=fileHandle)
        return  mae, mre

    def compute_score_byInterval(self,item,interval=1000):
        #TODO 本函数用于计算各个时间段内目标插入值的性能，以interval为样本点间隔
        # 比较 self.test_index 对应值与self.originaldata中对应数值之间的mse,
        #填充估计值
        for row in self.test_index.index:
            if row%5000==0:
                print('Process')
            list_index = self.originaldata.get_listIndex_by_id(self.test_index.loc[row,'station_id'])
            check_row = self.test_index.loc[row,'row']
            self.test_index.loc[row,'esti_data'] = self.originaldata.list[list_index].loc[check_row,item]
        self.test_index.to_csv(self.saveFilePath + 'test_score.csv')
        mae = []
        mre = []
        each_interval_datasize = []
        size = int(np.ceil(len(self.originaldata.list[0])/interval))#TODO 定义数据的区段数
        for i in range(size):
            start_index = i*interval
            end_index = (i+1)*interval
            if end_index >= self.datalength:
                end_index = self.datalength-1
            subdf_ = self.test_index[(self.test_index['row']>=start_index) & (self.test_index['row']< end_index)].copy()
            reald = subdf_['data'].values
            estid = subdf_['esti_data'].values
            mae.append(getMAE(reald, estid))
            mre.append(getMRE(reald, estid))
            each_interval_datasize.append(len(subdf_))
        reald_all = self.test_index['data'].values
        estid_all = self.test_index['esti_data'].values
        mae_all = getMAE(reald_all, estid_all)
        mre_all = getMRE(reald_all, estid_all)
        with open('result/output_result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
            # print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
            for i in range(size):
                print('IMA:','Range from',i*interval,'to',(i+1)*interval,'datasize=',each_interval_datasize[i],'MAE', mae[i], 'MRE', mre[i], file=fileHandle)
            print('IMA:','Range from begin to end','MAE', mae_all, 'MRE', mre_all, file=fileHandle)
        return mae, mre,mae_all,mre_all

    def run(self):
        #TODO 训练xgboost树模型
        self.train_xgboost_model()
        #TODO 解决blockmissing 数据插值
        self.block_missing_op()
        # TODO 补全全部的缺失数据 xgboost + mul-viewer
        imputed_file_name = self.saveFilePath+'mu_filled_file.csv'
        imputed_file_name_path = Path(imputed_file_name)
        if imputed_file_name_path.exists():
            self.originaldata = df_clip(pd.read_csv(imputed_file_name_path,index_col='Unnamed: 0'),'station_id')
        else:
            for param_index, params in enumerate(self.testItem, start=0):
                #TODO params:表示属性名称 param_index： 表示数量的索引，适用于对应训练模型的列表索引
                # column_index：表示该属性在原数据中的列索引
                # TODO 每个站点数据训练回归模型
                self.train_LinearRegression_model(param_index, params) #TODO LR模型为二位数据，第一维为属性，第二位为站点
                self.multi_viewer_imputer(params,self.columns[param_index],param_index)
            df_ = self.originaldata.get_mergedf()
            df_.to_csv(imputed_file_name)
        final_file_path = self.saveFilePath+'final_file.csv'
        final_file_path_path = Path(final_file_path)
        if final_file_path_path.exists(): #如果最终插值文件存在
            #load final data
            self.originaldata = df_clip(pd.read_csv(final_file_path_path,index_col='Unnamed: 0'),'station_id')
        else: #否则，则核对
            self.final_check()
            df_ = self.originaldata.get_mergedf()
            df_.to_csv(final_file_path_path)

    def run_data_with_category(self):
        #TODO 当数据类型中存在类别时，使用xgboost分类器
        pass

def get_pearson_neighbor(df,dis_df,item,outputfile):  #2017-2018
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
    file = 'origin/'+outputfile +item+ 'pearson_neighbor.csv'
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
    file = 'origin/'+outputfile +item+  'pearson_neighbor_dis.csv'
    neighbor_dis.to_csv(file)
    return neighbor_df, neighbor_dis
# def st_viewer(oped_data, indexdf, item, outfile, neighbordf, neighbor_dis_df, knnVal=-1, wind=7):
    #oped_data:删除测试位置的数据集，indexdf 测试机的位置,
    #item: 插入值列名
    #outfile:输出数据名称， neighbordf：邻域表
    #neighbor_dis_df 领域距离表
    #knnVal：K近邻的数量，取-1时表示选择全部邻居
    #wind:时间窗口的大小
    #check block loss
    # return 0,0
def run_test(subitem, year, outfile,interval=1000):
    with open('result/output_result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
        # print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
        print('Running test for ', year, ' ', subitem, file=fileHandle)
    # TODO outfile demo: 17_18_PM2_5
    if year == '14_15':
        # data_path = "../000data_pre_analyze/beijing20142015/beijing14_15_aq.csv"
        loss_log_output_path = 'origin/checkloss14_15_aq'+subitem+'.csv'
        dis_file = "../000data_pre_analyze/beijing20142015/14_15aq_aq_dis.csv"
        pearson_neighbor_file = 'origin/'+year + subitem + 'pearson_neighbor.csv'
        # pearson_neighbor_dis_file = 'origin/'+year + item + 'pearson_neighbor_dis.csv'
        pearson_neighbor_dis_file = '../000data_pre_analyze/beijing20142015/14_15aq_aq_dis.csv'
    elif year == '17_18':
        # data_path = "../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv"
        loss_log_output_path = 'origin/checkloss17_18_aq'+subitem+'.csv'
        dis_file = "../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv"
        pearson_neighbor_file ='origin/'+ year + subitem + 'pearson_neighbor.csv'
        # pearson_neighbor_dis_file ='origin/' + year + item + 'pearson_neighbor_dis.csv'
        pearson_neighbor_dis_file ='../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv'
    elif year =='1802_1803':
        # data_path = "../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv"
        loss_log_output_path = 'origin/checkloss1802_1803_aq' + subitem + '.csv'
        dis_file = "../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv"
        pearson_neighbor_file = 'origin/' + year + subitem + 'pearson_neighbor.csv'
        # pearson_neighbor_dis_file ='origin/' + year + item + 'pearson_neighbor_dis.csv'
        pearson_neighbor_dis_file = '../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv'
    else:
        assert 'Input right year: 14_15 or 17_18'


    #TODO 计算皮尔森相关系数领域
    temp_path = Path(pearson_neighbor_file)
    temp_path2 = Path(pearson_neighbor_dis_file)
    del_df = "origin/del_testdata_" + outfile + '.csv'
    if temp_path.exists() and temp_path2.exists():
        pass
    else: #计算
        oped_data = pd.read_csv(del_df)
        dis_df = pd.read_csv(dis_file, index_col='Unnamed: 0')
        neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data,dis_df,subitem,year)

    # del_df = "origin/del_testdata_" + outfile+'.csv'
    check_list = pd.read_csv(loss_log_output_path)
    # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    oped_data = pd.read_csv(del_df)

    # 计算pearson相关系数
    # # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data, dis_df,'PM2.5', 'origin/17_18PM2_5')

    neighbor_info = pd.read_csv(pearson_neighbor_file,  # 邻居信息
                                index_col='Unnamed: 0')
    neighbor_info = neighbor_info.sort_index()  # 重新排序index
    neighbor_dis_info = pd.read_csv(pearson_neighbor_dis_file,  # 邻居距离信息
                                    index_col='Unnamed: 0')
    item = [ 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2' ]
    testItem = [subitem]
    st_d = IMA(oped_data, check_list, item, testItem, 'auxiliaryFile/'+outfile+'/', neighbor_info,
                  neighbor_dis_info, knnVal=7, wind=11)
    st_d.run()
    # st_d.compute_score(subitem)
    st_d.compute_score_byInterval(subitem,interval)


if __name__ == '__main__':
    #1 compareomputingMethods.py check_loss
    #2 delete test_index data
    #3 insert method.
    #4 compute result

    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    #
    # #计算pearson相关系数
    # # # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data, dis_df,'PM2.5', 'origin/17_18PM2_5')
    #
    # neighbor_info = pd.read_csv("../ImputerOp/origin/17_18PM2.5pearson_neighbor.csv", #邻居信息
    #                             index_col='Unnamed: 0')
    # neighbor_info = neighbor_info.sort_index()  #重新排序index
    # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", #邻居距离信息
    #                                 index_col='Unnamed: 0')
    # item = [ 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2' ]
    # st_d = IMA(oped_data, check_list, item, 'auxiliaryFile/17_18/',neighbor_info,
    #               neighbor_dis_info,knnVal=7,wind=11)
    # st_d.run()
    # st_d.compute_score()
    #mae, mre = st_viewer(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv',neighbor_info, neighbor_dis_info,knnVal=-1,wind=7)
    #mae, mre = insert_multi_viewer_estimate(oped_data, check_list, 'PM2.5', '17_18_PM2_5.csv',neighbor_info, neighbor_dis_info,knnVal=-1,wind=7)
    #print(mae, mre)
    #TODO bk
    # check_list = pd.read_csv("origin/checkloss17_18_aq.csv")
    # # data = pd.read_csv("../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv")
    # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    #
    # # 计算pearson相关系数
    # # # oped_data = pd.read_csv("origin/del_testdata_17_18_PM2_5.csv")
    # # dis_df = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv", index_col='Unnamed: 0')
    # # neighbor_info, neighbor_dis_info = get_pearson_neighbor(oped_data, dis_df,'PM2.5', 'origin/17_18PM2_5')
    #
    # neighbor_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_neighbor.csv",  # 邻居信息
    #                             index_col='Unnamed: 0')
    # neighbor_info = neighbor_info.sort_index()  # 重新排序index
    # neighbor_dis_info = pd.read_csv("../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv",  # 邻居距离信息
    #                                 index_col='Unnamed: 0')
    # item = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    # st_d = IMA(oped_data, check_list, item, 'auxiliaryFile/17_18/', neighbor_info,
    #               neighbor_dis_info, knnVal=7, wind=11)
    # st_d.run()
    # st_d.compute_score()

    #17_18年的数据插值
    run_test('NO2', '17_18', '17_18_NO2',interval=2000)
    run_test('PM2.5', '17_18', '17_18',interval=2000)
    # #18 02-03的数据插值
    run_test('NO2', '1802_1803', '1802_1803_NO2_no_resi',interval=1000)
    run_test('PM2.5', '1802_1803', '1802_1803_PM2.5',interval=1000)


    # run_test('PM2.5', '14_15', '14_15_PM2.5', interval=2000)
    # run_test('NO2', '14_15', '14_15_NO2', interval=2000)