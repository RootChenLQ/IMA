import pandas as pd
import numpy as np
from pathlib import Path
from utils.Structure import df_clip
#将时间序列
def get_dateSeries(df):
    df['utc_time'] = pd.to_datetime(df['utc_time'])  #设置时间变量类型为datetime
    max_time = df['utc_time'].max()  #获取最大值
    # print(max_time)
    min_time = df['utc_time'].min() #获取最小值
    # print(min_time)
    date_index = pd.date_range(min_time,max_time,freq='H')
    print(date_index[0],'-',date_index[-1],'size:',len(date_index))
    return date_index

def complete_df_rows(df,outputFilename,item):
    #TODO 该函数将数据时刻缺失填充完整
    #time 标记 utc_time
    assert 'utc_time' in df.columns, 'check if utc_time exit in dataframe' #判断是否存在utc_time 变量
    #df['utc_time'] = pd.to_datetime(df['utc_time'])
    # print(df['utc_time'])
    #df['utc_time'] = pd.to_datetime(df['utc_time'])  #将时间变量设置为时间类型
    # print(df['utc_time'])
    date_series = get_dateSeries(df)   ##根据时间的最大最小值，产生间隔为1小时的时序
    ids = df['station_id'].unique() #获取station_id列表
    print('ids',ids)
    new_df = pd.DataFrame(columns=df.columns) #空的dataframe变量，用于存储完整的数据集
    # print(new_df)
    #设置两个变量的datetime为index
    for id in ids:
        print(id)  #打印id
        #temp_df = date_index.copy()
        subdf = df[df['station_id']==id].copy() #获取子集
        subdf = subdf.drop_duplicates(subset=['utc_time'], keep='first')  #删除时间重复的
        subdf['utc_time'] = pd.to_datetime(subdf['utc_time'])
        #subdf = subdf.set_index('utc_time')     #设置时间为索引
        temp_df = pd.DataFrame(columns=df.columns) #创建新的dataframe
        temp_df['utc_time'] = date_series       #填充时间序列
        #temp_df = temp_df.set_index('utc_time') #设置时间为索引
        for row in temp_df.index:             #获取时间
            if row %1000 ==0:
                print('Process；',row,'/',len(temp_df))
            # print(temp_df.loc[row,:])
            # print(subdf[subdf['utc_time']==temp_df.loc[row,'utc_time']])
            df_ = subdf[subdf['utc_time']==temp_df.loc[row,'utc_time']]
            if len(df_) > 0:
                temp_df.loc[row,:]=df_.loc[df_.index[0],:]
            else:
                temp_df.loc[row,'station_id'] = id
            # if date_ in subdf['utc_time']:               # 表示存在该数据
            #      temp_df.loc[date_,:] = subdf.loc[date_,:] #填充存在的数据行
            # else:
            #     # print(index)
            #     temp_df.loc[date_,'station_id'] = id   #填充id
        new_df = pd.concat([new_df,temp_df])           #拼接dataframe

    new_df.to_csv(outputFilename)
    return new_df
def _clean_outlier(df_,item):
    #TODO 当数据存在离群值异常值，则提出，后续插值方法进行插值
    id = df_.loc[0,'station_id']
    count = 0
    mean = np.mean(df_[item])
    limit = [] #获取上90%的位置的数值
    for item_ in item:
        # print(df_[item_])
        limit.append(np.nanpercentile(df_[item_],90))
    limit = np.array(limit)
    limit = limit * 3
    #TODO 获取属性对应数据的列index
    column_list = []
    for item_ in item:
        for i, name in enumerate(df_.columns,start=0):
            if name == item_:
                column_list.append(i)
                break

    for row in df_.index: #遍历全部索引
        if row %3000 ==0:
            print(id,row)
        for i, item_ in enumerate(item,start=0): #遍历全部属性
            start = row - 3
            end = row + 4
            if start < df_.index[0]:
                start = df_.index[0]
            if end > df_.index[-1]:
                end = df_.index[-1]
            index_list = [j for j in range(start,end) if j!=row]
            series = df_.iloc[index_list,column_list[i]:column_list[i] + 1] #获取除目标index外的数据
            nan_num = series.isna().sum().values[0]  # 该列nan值的数量
            nan_rate = nan_num / len(series)  # nan值的比例
            # if np.isnan(df_.loc[row,item_]) or nan_rate >2/3:
            if np.isnan(df_.loc[row,item_]) or nan_rate >2/3:
                pass
            else:
                if len(series)==0:
                    mean = np.nan
                else:
                    mean = np.nanmean(series.values)
                # std = np.nanstd(series.values)
                # print(df_.loc[row,:])
                value =df_.loc[row,item_]
                # mean_ = mean[i]
                thres = limit[i]
                if value -mean > limit[i]:
                    print(row)
                    df_.loc[row, item_] = np.nan
                    count += 1
    print('change times',count)
    return df_

def cleanAq_by_id(df,item,outputFilename):
    #TODO 根据id号清洗数据集中明显离群值
    new_df = pd.DataFrame(columns=df.columns)  # 空的dataframe变量，用于存储完整的数据集
    ids = df['station_id'].unique() #获取station_id列表
    for id in ids:
        subdf = df[df['station_id'] == id].copy()  # 获取子集
        subdf = subdf.reset_index(drop=True) #
        tempdf = _clean_outlier(subdf,item)
        new_df = pd.concat([new_df, tempdf])  # 拼接dataframe

    new_df.to_csv(outputFilename)
    return new_df

def cleanMeo_by_id(df,item,outputFilename):
    #TODO 根据id号清洗数据集中明显离群值
    new_df = pd.DataFrame(columns=df.columns)  # 空的dataframe变量，用于存储完整的数据集
    ids = df['station_id'].unique() #获取station_id列表
    for id in ids:
        subdf = df[df['station_id'] == id].copy()  # 获取子集
        subdf = subdf.reset_index(drop=True) #
        tempdf = _clean_outlier(subdf,item)
        new_df = pd.concat([new_df, tempdf])  # 拼接dataframe
    # TODO 读取文件 将14-15的文件中的数值型id改为str

    new_df.to_csv(outputFilename)

def aq_clean():
    #TODO 本函数用于2014-2015，20172018，20180203数据集的时间补齐，异常提出。
    aq_item = [ 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2' ]
    #将17-18的污染数据缺失时间行补齐
    data = pd.read_csv("../000original_data/beijing20172018/beijing_17_18_aq.csv")
    #clean_outlier(data, aq_item)
    outputFilename = "beijing20172018/beijing17_18_aq_filled_time.csv"
    outputFilenamePath = Path(outputFilename)
    if not outputFilenamePath.exists():
        complete_df_rows(data,outputFilename,aq_item)
    delete_outlier_filename = "beijing20172018/beijing17_18_aq.csv"
    data = pd.read_csv(outputFilename)
    cleanAq_by_id(data,aq_item,delete_outlier_filename)

    #TODO 将201802-201803的污染数据缺失时间补齐
    aq_item = [ 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2' ]
    data = pd.read_csv("../000original_data/beijing20172018/beijing_201802_201803_aq.csv")
    outputFilename = "beijing20172018/beijing201802_201803_aq_filled_time.csv"
    outputFilenamePath = Path(outputFilename)
    if not outputFilenamePath.exists():
        complete_df_rows(data, outputFilename, aq_item)
    delete_outlier_filename = "beijing20172018/beijing201802_201803_aq.csv"
    data = pd.read_csv(outputFilename)
    cleanAq_by_id(data, aq_item, delete_outlier_filename)

    #TODO #将14-15空气污染数据缺失时间戳补齐
    aq_item = [ 'PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2' ]
    data = pd.read_csv("../000original_data/beijing20142015/beijing_14_15_aq.csv")
    outputFilename = "beijing20142015/beijing14_15_aq_filled_time.csv"
    outputFilenamePath = Path(outputFilename)
    if not outputFilenamePath.exists():
        complete_df_rows(data, outputFilename,aq_item)
    delete_outlier_filename = "beijing20142015/beijing14_15_aq.csv"
    data = pd.read_csv(outputFilename)
    data = cleanAq_by_id(data, aq_item, delete_outlier_filename) #清除离群值
    #data = pd.read_csv(delete_outlier_filename)
    data['station_id'] = data['station_id'].astype(str) #设置stationid为str型
    data['station_id'] = 'A'+data['station_id'] #添加字母
    data.to_csv(delete_outlier_filename,index=False) #输出
    print(data.head())


def meo_clean():
    #TODO meo数据包含离散型，和类别类型
    # meo_item = ['temperature', 'pressure', 'wind_speed']
    # TODO 将17-18的气象数据缺失时间行补齐
    meo_item = ['temperature', 'pressure', 'wind_speed']
    data = pd.read_csv("../000original_data/beijing20172018/beijing_17_18_meo.csv")
    outputFilename = "beijing20172018/beijing17_18_meo_filled_time.csv"
    outputFilenamePath = Path(outputFilename)
    if not outputFilenamePath.exists():
        complete_df_rows(data, outputFilename, meo_item)
    delete_outlier_filename = "beijing20172018/beijing17_18_meo.csv"
    data = pd.read_csv(outputFilename)
    cleanMeo_by_id(data, meo_item, delete_outlier_filename)

    # TODO 将201802-201803的气象数据缺失时间补齐
    meo_item = ['temperature', 'pressure', 'wind_speed']
    data = pd.read_csv("../000original_data/beijing20172018/beijing_201802_201803_me.csv")
    outputFilename = "beijing20172018/beijing201802_201803_me_filled_time.csv"
    outputFilenamePath = Path(outputFilename)
    if not outputFilenamePath.exists():
        complete_df_rows(data, outputFilename, meo_item)
    delete_outlier_filename = "beijing20172018/beijing201802_201803_me.csv"
    data = pd.read_csv(outputFilename)
    cleanMeo_by_id(data, meo_item, delete_outlier_filename)

    #TODO #将14-15气象数据缺失时间戳数据补齐
    meo_item = ['temperature', 'pressure', 'wind_speed']
    data = pd.read_csv("../000original_data/beijing20142015/beijing_14_15_meo.csv")
    outputFilename = "beijing20142015/beijing14_15_meo_filled_time.csv"
    outputFilenamePath = Path(outputFilename)
    if not outputFilenamePath.exists():
        print(len(data))
        index_ = get_dateSeries(data)
        complete_df_rows(data,outputFilename,meo_item)
    delete_outlier_filename = "beijing20142015/beijing14_15_meo.csv"
    data = pd.read_csv(outputFilename)
    data = cleanMeo_by_id(data, meo_item, delete_outlier_filename)
    data = pd.read_csv(delete_outlier_filename)
    data['station_id'] = data['station_id'].astype(str)  # 设置stationid为str型
    data['station_id'] = 'M' + data['station_id']  # 添加字母
    data.to_csv(delete_outlier_filename, index=False)  # 输出
    print(data.head())

if __name__ == '__main__':
    #TODO 污染数据处理 done
    aq_clean()
    #TODO气象数据的插值
    meo_clean()