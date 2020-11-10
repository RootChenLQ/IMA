import math
import time
import datetime
import pandas as pd
import numpy as np
combine_data = [
    ## y:'PM2.5'; other: input
    'name','date','time',   #date , time -> embeded 
    'temperature','pressure','humidity','wind_direction','wind_speed','weather', #weather -> embeded
    'one_hour_later_PM2.5','PM10','NO2','CO','O3','SO2',
    'AQI1','AQI2','AQI3','AQI4','AQI5','AQI6','AQI7','AQI8',
    'AQI9','AQI10','AQI11','AQI12','AQI13','AQI14','AQI15','AQI16','AQI17']

#坐标名，经度， 维度
meo_info = ['name','longitude','latitude'] #编号 0，1，2， 35*3
aq_info = ['name','longitude','latitude']  #编号 0，1，2   18*3
#aq_aq_dis = [nxn]矩阵 检测站之间的距离
#aq_meo_dis = [nxm]矩阵 监测站和气象站的距离
Radius1 = 30 # 10KM 30KM
Radius2 = 100 #100KM
Log_unit = 111     #经度1度对于的距离
Lat_unit = 111 * math.cos(40)   #纬度1度对应度的距离
DataSize = 311010
#sunny clear 天晴 haze 霾 snow 雪 fog 雾 rain 下雨 dust 灰尘 sand 沙尘  sleet 雨夹雪  Rain/Snow with Hail 雨夹雪 Rain with Hail 雨加冰雹
# weather = {'Sunny/clear':0, 'Haze':1, 'Snow':2, 'Fog':3, 'Rain':4, 'Dust':5, 'Sand':6, 'Sleet':7,
#  'Rain/Snow with Hail':8, 'Rain with Hail':9}
weather = {'Sunny/clear':0,'Fog':1, 'Rain':2, 'Snow':3,'Sleet':4, 'Rain/Snow with Hail':5, 'Rain with Hail':6, 'Haze':7, 'Dust':8,'Sand':9}

meo_dic = {'shunyi_meo': 0, 'hadian_meo': 1, 'yanqing_meo': 2, 'miyun_meo': 3, 'huairou_meo': 4, 'shangdianzi_meo': 5, 
'pinggu_meo': 6, 'tongzhou_meo': 7, 'chaoyang_meo': 8, 'pingchang_meo': 9, 'zhaitang_meo': 10, 'mentougou_meo': 11, 
'beijing_meo': 12, 'shijingshan_meo': 13, 'fengtai_meo': 14, 'daxing_meo': 15, 'fangshan_meo': 16, 'xiayunling_meo': 17}

aq_dic = {'aotizhongxin_aq': 0, 'badaling_aq': 1, 'beibuxinqu_aq': 2, 'daxing_aq': 3, 'dingling_aq': 4, 'donggaocun_aq': 5,
 'dongsi_aq': 6, 'dongsihuan_aq': 7, 'fangshan_aq': 8, 'fengtaihuayuan_aq': 9, 'guanyuan_aq': 10,
          'gucheng_aq': 11,'huairou_aq': 12, 'liulihe_aq': 13, 'mentougou_aq': 14, 'miyun_aq': 15,
          'miyunshuiku_aq': 16, 'nansanhuan_aq': 17, 'nongzhanguan_aq': 18, 'pingchang_aq': 19, 'pinggu_aq': 20,
          'qianmen_aq': 21, 'shunyi_aq': 22, 'tiantan_aq': 23, 'tongzhou_aq': 24, 'wanliu_aq': 25,
          'wanshouxigong_aq': 26, 'xizhimenbei_aq': 27, 'yanqin_aq': 28, 'yizhuang_aq': 29, 'yongdingmennei_aq': 30,
          'yongledian_aq': 31, 'yufa_aq': 32, 'yungang_aq': 33, 'zhiwuyuan_aq': 34}

aq_id = ['aotizhongxin_aq', 'badaling_aq', 'beibuxinqu_aq', 'daxing_aq', 'dingling_aq', 'donggaocun_aq',
 'dongsi_aq', 'dongsihuan_aq', 'fangshan_aq', 'fengtaihuayuan_aq', 'guanyuan_aq', 'gucheng_aq',
'huairou_aq', 'liulihe_aq', 'mentougou_aq', 'miyun_aq', 'miyunshuiku_aq', 'nansanhuan_aq',
'nongzhanguan_aq', 'pingchang_aq', 'pinggu_aq', 'qianmen_aq', 'shunyi_aq', 'tiantan_aq',
'tongzhou_aq', 'wanliu_aq', 'wanshouxigong_aq', 'xizhimenbei_aq', 'yanqin_aq', 'yizhuang_aq',
'yongdingmennei_aq', 'yongledian_aq', 'yufa_aq', 'yungang_aq', 'zhiwuyuan_aq']

class df_clip():
    # 按照column name 分解 df -> df array
    def __init__(self,df,clip_by_column):
        # assert 'station_id' in df.columns
        self.columns = list(df.columns)
        # del self.columns[0]  #20200617
        if 'utc_time' in df.columns: # for those time series data
            df['utc_time'] = pd.to_datetime(df['utc_time'])
            #获取stationid的集合
            self.id = df[clip_by_column].unique()
            index = [i for i in range(len(self.id))]
            # dictionary
            self.dic = dict(zip(self.id, index))
            self.list = [pd.DataFrame() for i in range(len(self.id))]
            # self.mean_ = [pd.DataFrame() for i in range(len(self.id))]
            self.max_ = np.max(df)
            self.min_ = np.min(df)
            self.mean = np.mean(df[['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2' ]])
            for id_ in self.dic:
                val = self.dic[id_]  # 根据id,建立list数组
                self.list[val] = df[df[clip_by_column] == id_].copy()
                self.list[val] = self.list[val][self.columns]  # 将id去除
                # 2020 0622 add
                self.list[val].sort_values(['utc_time'])
                self.list[val] = self.list[val].reset_index(drop=True)
                # 2020 0623
                # print(self.list[val].iloc[:, 2:])
                # mean_arr = np.nanmean(self.list[val].iloc[:, 3:].values, 0)
                #self.mean_[val] = self.list[val].iloc[:, 3:].mean(axis=0)
        else:
            self.id = df[clip_by_column].unique()
            index = [i for i in range(len(self.id))]
            # dictionary
            self.dic = dict(zip(self.id, index))
            self.list = [pd.DataFrame() for i in range(len(self.id))]
            self.mean_ = [pd.DataFrame() for i in range(len(self.id))]
            for id_ in self.dic:
                val = self.dic[id_]  # 根据id,建立list数组
                self.list[val] = df[df[clip_by_column] == id_].copy()
                self.list[val] = self.list[val][self.columns]  # 将id去除
                # 2020 0623
                # print(self.list[val].iloc[:, 2:])
                # mean_arr = np.nanmean(self.list[val].iloc[:, 3:].values, 0)
                self.mean_[val] = self.list[val].iloc[:, 3:].mean(axis=0)

        # if 'utc_time' in df.columns: # for those time series data
        #     df['utc_time'] = pd.to_datetime(df['utc_time'])
        # #获取stationid的集合
        # self.id = df[clip_by_column].unique()
        # index = [i for i in range(len(self.id))]
        # #dictionary
        # self.dic = dict(zip(self.id ,index))
        # self.list = [pd.DataFrame() for i in range(len(self.id))]
        # self.mean_ = [pd.DataFrame() for i in range(len(self.id))]
        # for id_ in self.dic:
        #     val = self.dic[id_]  #根据id,建立list数组
        #     self.list[val] = df[df[clip_by_column]==id_].copy()
        #     self.list[val] = self.list[val][self.columns]  #将id去除
        #     #2020 0622 add
        #     self.list[val].sort_values(['utc_time'])
        #     self.list[val]= self.list[val].reset_index(drop=True)
        #     # 2020 0623
        #     #print(self.list[val].iloc[:, 2:])
        #     # mean_arr = np.nanmean(self.list[val].iloc[:, 3:].values, 0)
        #     self.mean_[val] = self.list[val].iloc[:, 3:].mean(axis = 0)
            # self.mean_[val] = pd.DataFrame(mean_arr,
            #                                columns = self.list[val].columns[3:]
            #                                )
            #
    def get_df_by_name(self,name,item,val):
        if not isinstance(val, np.datetime64):
            val = pd.to_datetime(val)
        index = self.dic[name]
        temp = self.list[index]
        #df_ = temp[temp[item] == val].iloc[0:-1] #只保留第一行数据
        df_ = temp[temp[item] == val]
        if len(df_) > 1: #存在数据重复问题
            df_ = df_.iloc[0:-1]
        if df_.empty:
            return pd.DataFrame() 
        else:
            return df_
    def get_df_by_id(self,id):
        #id 文件名
        
        if id in self.id:
            index = self.dic[id]
            df_ = self.list[index]
            return df_
        else:
            return pd.DataFrame()

    # def get_listIndex_by_id(self, name):
    #     assert name in self.id, "The queried station_in is not in DataFrame, check the spell"
    #     index = self.dic[name]
    #     return index
    def get_listIndex_by_id(self, name):
        #TODO
        if name not in self.id:
            return -1
        else:
            index = self.dic[name]
            return index
    def get_query(self,name,query_str):
        val = self.dic[name]
        df_ = self.list[val].query(query_str)
        return df_

    def test_search(self,name,time):
        t = time.time()
        print(time.time()-t)

    def get_mergedf(self):
        first = True
        for df_ in self.list:
            if first:
                mergedf = df_
                first = False
            else:
                mergedf = pd.concat([mergedf,df_])
        mergedf = mergedf.reset_index(drop = True)
        return mergedf

if __name__ == "__main__":
    #test onehot
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder 
    import torch
    # testdata = pd.DataFrame({'pet': ['chinese', 'english', 'english', 'math'],
                        #  'age': [6 , 5, 2, 2],
                        #  'salary':[7, 5, 2, 5]})
    # result  = OneHotEncoder(sparse = False).fit_transform(testdata[['pet']])
    # print(result)
    # '''
    #     station ID  的one-hot编码
    # '''
    # position = pd.read_csv('KDD2018Original/beijing_AirQuality_Station.csv',sep=',')
    # # print(position)
    # ID_one_df = pd.get_dummies(position['stationName'])
    # # print(ID_one_df)
    # tensor_df = torch.tensor(ID_one_df.values).float().t()
    # # print(tensor_df)
    # # print(tensor_df.shape)
    # embedd_weight = torch.randn(35,3).float()
    # embed_data = torch.mm(tensor_df, embedd_weight) # 获取one-hot编码后







    # '''
    #     utc_time  的计算mounth， week，time of day (0,1,2,3)
    # '''
   
    # #空气污染
    # date = pd.read_csv('KDDCleared/imputed_beijing_17_18_aq.csv',sep=',')
    # print(date.head())
    # datetime = date['utc_time']

    # datetime_ = date[0:4]
    # #/2018/1/31 16:00:00
    # # datetime_ =  pd.to_datetime(datetime_)
    # # print(datetime_[0])
    
    # from datetime import datetime
    # import numpy as np
    # # time = datetime.strptime(datetime_['utc_time'][0],"%Y-%m-%d %H:%M:%S")
    # # mounth = time.month    #获取日期中的月份
    # # week = time.weekday()  #获取日期中的星期数
    # # hour = time.hour       #获取日期中的小时
    # # print('mounth',mounth,' ', 'week',week,' ','hour',hour)
  
    # datetime_['mounth'] =  np.zeros(datetime_.shape[0])
    # datetime_['week'] =  np.zeros(datetime_.shape[0])
    # datetime_['hour'] = np.zeros(datetime_.shape[0])
    # print(datetime_)
    # for i in range(len(datetime_)):
    #     time  = datetime.strptime(datetime_.at[i,'utc_time'],"%Y-%m-%d %H:%M:%S")
    #     datetime_.at[i,'mounth'] =  time.month
    #     datetime_.at[i,'week'] = time.weekday()
    #     datetime_.at[i,'hour'] =  np.floor(time.hour /6 )
    # print(datetime_)
    # '''
    # print(embed_data)
    # print(ID_one_df)
    # tensor_one = torch.tensor(ID_one_df['aotizhongxin_aq'].values)
    # tensor_one = tensor_one.view(1,35).float()
    # embed_data_one = torch.mm(tensor_one, embedd_weight)
    # print(embed_data_one)
    # '''
    # '''
    # #将一个series数组转化为embed 输出
    # ID_one_df = pd.get_dummies(position['stationName'])
    # print(ID_one_df.shape)
    # print(ID_one_df['aotizhongxin_aq'].to_numpy().reshape(1,35))
    # #坐标的转化，即将
    # import torch
    # tensor_id = torch.tensor(ID_one_df['aotizhongxin_aq'].values)
    # print(tensor_id.view(1,35))
    # tensor_id = tensor_id.view(1,35).float()
    # print(tensor_id.shape)
    # embed = torch.randn(35,3).float()
    # print(embed)
    # embed_data = torch.mm(tensor_id, embed)
    # print(embed_data)
    # '''
    
    
    '''
    训练监测站 信息
    '''
    # date = pd.read_csv('KDDCleared/imputed_beijing_17_18_aq.csv',sep=',')
    # meo_pos = date[['station_id']]
    # # print(meo_pos['station_id'].unique())
    #
    # meo_pos = meo_pos.drop_duplicates(subset=['station_id'], keep='first')
    # meo_pos = meo_pos.reset_index(drop=True)
    # print('train_aq',meo_pos)


    '''
    测试监测站 信息
    '''
   
    # date = pd.read_csv('KDDCleared/imputed_beijing_201802_201803_aq.csv',sep=',')
    # meo_pos = date[['station_id']]
    # # print(meo_pos['station_id'].unique())
    # meo_pos = meo_pos.drop_duplicates(subset=['station_id'], keep='first')
    # meo_pos = meo_pos.reset_index(drop=True)
    # print('test_aq',meo_pos)


    '''
    训练气象站 信息
    '''
    # date = pd.read_csv('KDD2018Original/beijing_17_18_meo.csv',sep=',')
    # meo_pos = date[['station_id','latitude','longitude']]
    # # print(meo_pos['station_id'].unique())
    #
    # meo_pos = meo_pos.drop_duplicates(subset=['station_id'], keep='first')
    # meo_pos = meo_pos.reset_index(drop=True)
    # print('train_meo',meo_pos)
    # meo_pos.to_csv('meo_info.csv',index=None)

    '''
    测试气象站 信息
    '''
    # date = pd.read_csv('KDD2018Original/beijing_201802_201803_me.csv',sep=',')
    # meo_pos = date[['station_id']]
    # print(meo_pos['station_id'].unique())
    #
    # meo_pos = meo_pos.drop_duplicates(subset=['station_id'], keep='first')
    # meo_pos = meo_pos.reset_index(drop=True)
    # print('test_meo',meo_pos)

    '''
    读取 meo_info 与aq_info的信息，求距离矩阵
    '''
    # meo_info =  pd.read_csv('meo_info.csv',sep=',')
    # print(meo_info.head())
    # aq_info = pd.read_csv('aq_info.csv',sep=',')
    # print(aq_info.head())
    #-》 compute_dis中
  
    '''
    读取 meo dic
    '''
    # import pandas as pd
    # meo_df = pd.read_csv('KDD2018Original/beijing_17_18_meo.csv',sep=',')
    # id_ = meo_df['station_id'].unique()
    # index = [i for i in range(len(id_))]
    # meo_dic = dict(zip(id_,index))
    # print(meo_dic)
    '''
    读取 aq dic
    '''
    # aq_df = pd.read_csv('KDDCleared/imputed_beijing_17_18_aq.csv',sep=',',index_col = 'Unnamed: 0')
    # id_ = aq_df['station_id'].unique()
    # index = [i for i in range(len(id_))]
    # aq_dic = dict(zip(id_,index))
    # print(aq_dic)
    '''
    读取weather dic 
    '''
    # meo_df = pd.read_csv('KDD2018Original/beijing_17_18_meo.csv',sep=',')
    # id_ = meo_df['weather'].unique()
    # index = [i for i in range(len(id_))]
    # weather_dic = dict(zip(id_,index))
    # print(weather_dic)

   
    # #test aq clip function
    # aq_df_clip = df_clip(aq_df,'station_id')
    # str = '2017/7/13 13:00:00'
    # time_ = datetime.datetime.strptime(str,"%Y/%m/%d %H:%M:%S")
    # t1 = time.time()
    # d1 = aq_df_clip.get_df_by_name('beibuxinqu_aq','utc_time',str)
    # print(d1)
    # print('stop')
    # print(time.time()-t1)
    #
    # t1 = time.time()
    # df_ = aq_df[aq_df['station_id']=='beibuxinqu_aq']
    # df_2 = df_[df_['utc_time']==time_]
    # print(df_2)
    # print('stop')
    # print(time.time()-t1)
    # 按照column name 分解 df -> df array

    ''' 
      # test get mean
    '''
    # date = pd.read_csv('KDDCleared/imputed_beijing_17_18_aq.csv', sep=',')
    # aq_df_clip = df_clip(date, 'station_id')
    # print(aq_df_clip.mean_[0]['PM2.5'])

    # test get merge data
    fileNames = ['KDD2018Original/beijing_17_18_aq.csv']
    df_1 = pd.read_csv(fileNames[0])
    df_byid = df_clip(df_1, 'station_id')
    df_2 = df_byid.get_mergedf()
    print(df_1.head())
    print(df_2.head())
    print('')