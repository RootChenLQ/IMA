import pandas as pd
import numpy as np
from utils import func

# from Structure import df_clip
'''
计算经纬度坐标之间的距离             def getdistance
计算以坐标1为圆形的坐标2所在的空间区域 def get_area
'''
# 坐标名，经度， 维度
meo_info = ['name', 'longitude', 'latitude']  # 编号 0，1，2， 35*3
aq_info = ['name', 'longitude', 'latitude']  # 编号 0，1，2   18*3
# aq_aq_dis = [nxn]矩阵 检测站之间的距离
# aq_meo_dis = [nxm]矩阵 监测站和气象站的距离

np.random.seed(11)


# 计算两个地理坐标间的距离

def get_aq_dis_file(aq_info,oufile):
    # aq_df为aq星系变量
    # df_ = aq_info.copy()
    aq_aq_dis = np.zeros((len(aq_info), len(aq_info)))
    aq_list = aq_info['station_id'].values
    for i in range(len(aq_info)):
        # 计算 aq_aq_dis
        for j in range(i, len(aq_info)):
            aq_aq_dis[i, j] = func.geodistance(aq_info.at[i, 'longitude'], aq_info.at[i, 'latitude'],
                                               aq_info.at[j, 'longitude'], aq_info.at[j, 'latitude'])
            aq_aq_dis[j, i] = aq_aq_dis[i, j]
    # print(aq_aq_dis)
    # 填充数值
    df_ = pd.DataFrame(aq_aq_dis)
    # 设置横轴 和 列名
    df_.columns = aq_list
    df_ = df_.set_index(aq_list)
    # 输出文件
    # file = year + 'aq_aq_dis.csv'
    df_.to_csv(oufile)


def get_aq_meo_dis_file(aq_info, meo_info, oufile):
    aq_meo_dis = np.zeros((len(aq_info), len(meo_info)))
    # 获取aq meo的id
    aq_list = aq_info['station_id'].values
    meo_list = meo_info['station_id'].values
    for i in range(len(aq_info)):
        # 计算 aq_aq_dis
        for j in range(len(meo_info)):
            aq_meo_dis[i, j] = func.geodistance(aq_info.at[i, 'longitude'], aq_info.at[i, 'latitude'],
                                                meo_info.at[j, 'longitude'], meo_info.at[j, 'latitude'])
    # 填充数据
    df_ = pd.DataFrame(aq_meo_dis)
    # print(df_.shape)
    # 设置index 和列名
    df_.columns = meo_list
    df_ = df_.set_index(aq_list)
    # 输出文件
    # file = year + 'aq_meo_dis.csv'
    df_.to_csv(oufile)


def get_aq_area_file(aq_info,oufile):
    # aq_info = pd.read_csv('aq_info.csv',sep=',')
    # print(aq_info.head())
    from_aq = []  # 记录每一对aq-aq
    to_aq = []
    area = []
    lever = []
    for i in range(len(aq_info)):
        for j in range(len(aq_info)):
            # l,x,t =
            # aq_aq_dis[i,j] = getangle(aq_info.at[i,'longitude'],aq_info.at[i,'latitude'],aq_info.at[j,'longitude'],aq_info.at[j,'latitude'])
            from_aq.append(aq_info.at[i, 'station_id'])
            to_aq.append(aq_info.at[j, 'station_id'])
            area_, level_ = func.get_area(aq_info.at[i, 'longitude'], aq_info.at[i, 'latitude'],
                                          aq_info.at[j, 'longitude'], aq_info.at[j, 'latitude'])
            area.append(area_)
            lever.append(level_)
    # 创造multi index Dataframe
    multi_df = pd.DataFrame({
        'from_aq': from_aq,
        'to_aq': to_aq,
        'area': area,
        'level': lever}
    )
    multi_df.set_index(['from_aq', 'to_aq'], inplace=True)
    # print(multi_df[0:100])
    # file = year+'area.csv'
    multi_df.to_csv(oufile)


def get_aq_meo_neighbors(aq_meo_dis_df,oufile):
    # input: aq_meo_dis dataframe
    #
    index_ = aq_meo_dis_df.index
    columns_ = aq_meo_dis_df.columns
    # 获取数据数组
    data_array = aq_meo_dis_df.values
    # 置负值
    data_array = data_array * -1
    # print(data_array)
    # print(data_array.shape)
    aq_meo_dis_df = aq_meo_dis_df.reset_index(drop=True)
    # print(aq_meo_dis_df.head())
    new_columns = []
    for i in range(len(columns_)):  # 对df中全部行进行距离排序
        column_name = 'neighbor' + str(i + 1)
        new_columns.append(column_name)  # 保存新的列名
        aq_meo_dis_df[column_name] = data_array.argmax(axis=1)  # 获得最大值的位置(愿数据乘以-1操作，原来最小值即现在最大值)
        data_array[aq_meo_dis_df.index, np.argmax(data_array, axis=1)] = -1e10  # 给最大值置小值
    aq_meo_neighbor = aq_meo_dis_df[new_columns].copy()  # 提取插入的neighbor数据
    # 将内容存储为meo
    for i in range(len(columns_)):
        # print(i)
        aq_meo_neighbor = aq_meo_neighbor.replace(i, columns_[i])  # 将排列信息，转化为气象站的名字
    # print(aq_meo_neighbor.head())
    # 将index 设置为aq
    aq_meo_neighbor = aq_meo_neighbor.set_index(index_)
    # print(aq_meo_neighbor.head())
    # file = year+'aq_meo_neighbor.csv'
    aq_meo_neighbor.to_csv(oufile)


def get_aq_aq_neighbors(aq_aq_dis_df,oufile):
    # input: aq_aq dataframe
    # 根据aq到aq的distance(或者别的度量值),距离采用降序排序，别的度量根据具体的属性排序。
    index_ = aq_aq_dis_df.index  #行的index
    columns_ = aq_aq_dis_df.columns #列名
    # 获取数据数组
    data_array = aq_aq_dis_df.values #获取matrix的数值
    # 置负值
    data_array = data_array * -1  #将数值取反，最小值变为最大值
    # print(data_array)
    # print(data_array.shape)
    aq_aq_dis_df = aq_aq_dis_df.reset_index(drop=True)  #重新设置index,抛弃原始的index
    # print(aq_aq_dis_df.head())
    new_columns = []
    for i in range(len(columns_)):  # 对df中全部行进行距离排序
        column_name = 'neighbor' + str(i + 1)  #列名neighbor(i+1)
        new_columns.append(column_name)  # 保存新的列名
        aq_aq_dis_df[column_name] = data_array.argmax(axis=1)  # 获得最大值的位置(愿数据乘以-1操作，原来最小值即现在最大值)
        data_array[aq_aq_dis_df.index, np.argmax(data_array, axis=1)] = -1e10  # 给最大值置小值
    aq_aq_neighbor = aq_aq_dis_df[new_columns].copy()  # 提取插入的neighbor数据
    # 将内容存储为aq
    for i in range(len(columns_)):
        # print(i)
        aq_aq_neighbor = aq_aq_neighbor.replace(i, columns_[i])  # 将排列信息，转化为气象站的名字
    # print(aq_aq_neighbor.head())
    # 将index 设置为aq
    aq_aq_neighbor = aq_aq_neighbor.set_index(index_)   #重新设置 index为station_id
    # print(aq_aq_neighbor.head())
    # file = year + 'aq_aq_neighbor.csv'
    aq_aq_neighbor.to_csv(oufile)


def get_meo_dis_file(meo_info,oufile):
    meo_meo_dis = np.zeros((len(meo_info), len(meo_info)))
    meo_list = meo_info['station_id'].values
    for i in range(len(meo_info)):
        # 计算 aq_aq_dis
        for j in range(i, len(meo_info)):
            meo_meo_dis[i, j] = func.geodistance(meo_info.at[i, 'longitude'], meo_info.at[i, 'latitude'],
                                                 meo_info.at[j, 'longitude'], meo_info.at[j, 'latitude'])
            meo_meo_dis[j, i] = meo_meo_dis[i, j]
    # print(aq_aq_dis)
    # 填充数值
    df_ = pd.DataFrame(meo_meo_dis)
    # 设置横轴 和 列名
    df_.columns = meo_list
    df_ = df_.set_index(meo_list)
    # 输出文件
    # file = year + 'meo_meo_dis.csv'
    df_.to_csv(oufile)


def get_meo_meo_neighbors(meo_meo_dis_df,oufile):
    # input: aq_aq dataframe
    #
    index_ = meo_meo_dis_df.index
    columns_ = meo_meo_dis_df.columns
    # 获取数据数组
    data_array = meo_meo_dis_df.values
    # 置负值
    data_array = data_array * -1
    # print(data_array)
    # print(data_array.shape)
    meo_meo_dis_df = meo_meo_dis_df.reset_index(drop=True)
    # print(meo_meo_dis_df.head())
    new_columns = []
    for i in range(len(columns_)):  # 对df中全部行进行距离排序
        column_name = 'neighbor' + str(i + 1)
        new_columns.append(column_name)  # 保存新的列名
        meo_meo_dis_df[column_name] = data_array.argmax(axis=1)  # 获得最大值的位置(愿数据乘以-1操作，原来最小值即现在最大值)
        data_array[meo_meo_dis_df.index, np.argmax(data_array, axis=1)] = -1e10  # 给最大值置小值
    aq_aq_neighbor = meo_meo_dis_df[new_columns].copy()  # 提取插入的neighbor数据
    # 将内容存储为aq
    for i in range(len(columns_)):
        # print(i)
        aq_aq_neighbor = aq_aq_neighbor.replace(i, columns_[i])  # 将排列信息，转化为气象站的名字
    # print(aq_aq_neighbor.head())
    # 将index 设置为aq
    aq_aq_neighbor = aq_aq_neighbor.set_index(index_)
    # print(aq_aq_neighbor.head())
    # file = year + 'meo_meo_neighbor.csv'
    aq_aq_neighbor.to_csv(oufile)

def op_14_15_info():#处理北京17-18数据
    '''
           #1 读取 meo_info 与aq_info的信息，test reading
       '''
    # ############
    print('operation 1: test reading')
    meo_info = pd.read_csv('../000original_data/beijing20142015/meo_info1415.csv', sep=',')
    print(meo_info.head())
    aq_info = pd.read_csv('../000original_data/beijing20142015/aq_info1415.csv', sep=',')
    print(aq_info.head())
    # ############

    '''
    #2 #compute aq_aq_dis, 计算大气质量监测站之间的距离
       #output: aq_aq_dis.csv
    '''
    print('operation 2:compute aq_aq_dis')
    # ############
    get_aq_dis_file(aq_info, '14_15aq_aq_dis.csv')
    # ############

    '''
    #3 compute meo_meo_dis  new 0701
    # output meo_meo_dis.csv
    '''
    print('operation 3:compute meo_meo_dis ')
    get_meo_dis_file(meo_info, '14_15meo_meo_dis.csv')

    '''
    #4 #compute aq_meo_dis 计算监测站点与气象站点的距离
       #output: aq_meo_dis.csv
    '''
    # ############
    print('operation 4:compute aq_meo_dis')
    get_aq_meo_dis_file(aq_info, meo_info, '14_15aq_meo_dis.csv')
    # ############

    '''
    #5 read dis 读取aq_aq_dis.csv aq_meo_dis.csv文件
    #  index_col = 'Unnamed: 0' 第一列为index，默认读取名为“Unnamed: 0”
    '''
    # ############
    print('operation 5: test reading aq_aq_dis.csv aq_meo_dis.csv ')
    aq_aq_df = pd.read_csv('../000data_pre_analyze/14_15aq_aq_dis.csv', sep=',', index_col='Unnamed: 0')
    print(aq_aq_df.head())
    aq_meo_df = pd.read_csv('../000data_pre_analyze/14_15aq_meo_dis.csv', sep=',', index_col='Unnamed: 0')
    print(aq_meo_df.head())
    # ############

    # 计算aq坐标为圆心，其他aq坐标所在的区域
    # 使用pandas的Panel 存储三维数据， python2 可用
    # 第一位items - axis 0 , 每个item对应一个DataFrame
    # major_axis - axis 1，代表每个DataFrame的索引
    # minor_axis - axis 2, 代表每个DataFrame的列
    # pandas.Panel(data, items, major_axis, minor_axis, dtype, copy)
    # test function
    # get_area(11,20,10,20)
    '''
    #6 存储 area_info,lever
    '''
    # ############
    print('operation 6: compute area info')
    aq_info = pd.read_csv('../000original_data/beijing20142015/aq_info1415.csv', sep=',')
    get_aq_area_file(aq_info, '14_15area.csv')
    # ############

    '''
    #7 计算aq的neighbor https://blog.csdn.net/weixin_37536446/article/details/82774659
    '''
    # ############
    print('operation 7: compute aq_meo neighbor')
    aq_meo_df_ = pd.read_csv('../000data_pre_analyze/14_15aq_meo_dis.csv', sep=',', index_col='Unnamed: 0')
    get_aq_meo_neighbors(aq_meo_df_, '14_15aq_meo_neighbor.csv')
    # ############

    '''
    # # 读取area 文件
    # # multi_index dataframe的使用
    # '''
    # multi_df = pd.read_csv('TempData/area.csv',sep=',')
    # multi_df.set_index(['from_aq', 'to_aq'],inplace=True)
    # # print(multi_df.head())
    # # print(multi_df.shape)
    # # print(multi_df.head())
    # #利用df.query()来取数
    # from_aq_ = "yufa_aq"
    # to_aq_ = "liulihe_aq"
    # str_ =  'from_aq == \"'+from_aq_+'\"and to_aq ==\"'+to_aq_+"\""
    # df = multi_df.query(str_).copy()
    # # print(df['area'].values[0])
    # # print(multi_df.query('from_aq=="yufa_aq"'))
    # df_ = multi_df.query('from_aq=="yufa_aq" and level < 3 and level > 0').copy()
    # print(df_.sort_values(by = 'area'))
    # # aq 最近的meo信息
    # # 整合单个时刻多个数据，计算PM2.5的 AQIs
    # # 输出dataframe 包涵aq_id time pollutions meos(mean) aqis(pm2.5)
    # # day_of_weeek mount
    # aqis = np.zeros(17)
    # #平均PM2.5的浓度，计算分区中PM2.5的AQIs
    # print(aqis)
    # #直接获取某个area值
    # print(multi_df.loc[(from_aq_,to_aq_), 'area'])
    # #
    # print(multi_df.loc[(from_aq_,to_aq_), 'level'])

    ''' 
    # 8 compute aq_aq_neighbor_info
    '''
    aq_aq_df_ = pd.read_csv('../000data_pre_analyze/14_15aq_aq_dis.csv', sep=',', index_col='Unnamed: 0')
    get_aq_aq_neighbors(aq_aq_df_, '14_15aq_aq_neighbor.csv')
    '''
    #9 compute aq_aq_neighbor_info
    '''
    meo_meo_df_ = pd.read_csv('../000data_pre_analyze/14_15meo_meo_dis.csv', sep=',', index_col='Unnamed: 0')
    get_meo_meo_neighbors(meo_meo_df_, '14_15meo_meo_neighbor.csv')
def op_17_18_info():
    '''
        #1 读取 meo_info 与aq_info的信息，test reading
    '''
    # ############
    print('operation 1: test reading')
    meo_info = pd.read_csv('../000original_data/beijing20172018/meo_info.csv', sep=',')
    print(meo_info.head())
    aq_info = pd.read_csv('../000original_data/beijing20172018/aq_info.csv', sep=',')
    print(aq_info.head())
    # ############

    '''
    #2 #compute aq_aq_dis, 计算大气质量监测站之间的距离
       #output: aq_aq_dis.csv
    '''
    print('operation 2:compute aq_aq_dis')
    # ############
    # aq_info = pd.read_csv('TempData/aq_info.csv', sep=',')
    get_aq_dis_file(aq_info,'17_18aq_aq_dis.csv')
    # ############

    '''
    #3 compute meo_meo_dis  new 0701
    # output meo_meo_dis.csv
    '''
    # meo_info = pd.read_csv('TempData/meo_info.csv', sep=',')
    print('operation 3:compute meo_meo_dis ')
    get_meo_dis_file(meo_info,'17_18meo_meo_dis.csv')

    '''
    #4 #compute aq_meo_dis 计算监测站点与气象站点的距离
       #output: aq_meo_dis.csv
    '''
    # ############
    # meo_info = pd.read_csv('TempData/meo_info.csv', sep=',')
    # aq_info = pd.read_csv('TempData/aq_info.csv', sep=',')
    print('operation 4:compute aq_meo_dis')
    get_aq_meo_dis_file(aq_info, meo_info,'17_18aq_meo_dis.csv')
    # ############

    '''
    #5 read dis 读取aq_aq_dis.csv aq_meo_dis.csv文件
    #  index_col = 'Unnamed: 0' 第一列为index，默认读取名为“Unnamed: 0”
    '''
    # ############
    print('operation 5: test reading aq_aq_dis.csv aq_meo_dis.csv ')
    aq_aq_df = pd.read_csv('../000data_pre_analyze/17_18aq_aq_dis.csv', sep=',', index_col='Unnamed: 0')
    print(aq_aq_df.head())
    print(aq_aq_df.at['tongzhou_aq', 'dongsihuan_aq'])
    print(aq_aq_df.at['tongzhou_aq', 'tongzhou_aq'])
    aq_meo_df = pd.read_csv('../000data_pre_analyze/17_18aq_meo_dis.csv', sep=',', index_col='Unnamed: 0')
    print(aq_meo_df.head())
    print(aq_meo_df.at['liulihe_aq', 'hadian_meo'])
    # ############

    # 计算aq坐标为圆心，其他aq坐标所在的区域
    # 使用pandas的Panel 存储三维数据， python2 可用
    # 第一位items - axis 0 , 每个item对应一个DataFrame
    # major_axis - axis 1，代表每个DataFrame的索引
    # minor_axis - axis 2, 代表每个DataFrame的列
    # pandas.Panel(data, items, major_axis, minor_axis, dtype, copy)
    # test function
    # get_area(11,20,10,20)
    '''
    #6 存储 area_info,lever
    '''
    # ############
    print('operation 6: compute area info')
    aq_info = pd.read_csv('../000original_data/beijing20172018/aq_info.csv', sep=',')
    get_aq_area_file(aq_info,'17_18area.csv')
    # ############

    '''
    #7 计算aq的neighbor https://blog.csdn.net/weixin_37536446/article/details/82774659
    '''
    # ############
    print('operation 7: compute aq_meo neighbor')
    aq_meo_df_ = pd.read_csv('../000data_pre_analyze/17_18aq_meo_dis.csv', sep=',', index_col='Unnamed: 0')
    get_aq_meo_neighbors(aq_meo_df_,'17_18aq_meo_neighbor.csv')
    # ############

    '''
    # # 读取area 文件
    # # multi_index dataframe的使用
    # '''
    # multi_df = pd.read_csv('TempData/area.csv',sep=',')
    # multi_df.set_index(['from_aq', 'to_aq'],inplace=True)
    # # print(multi_df.head())
    # # print(multi_df.shape)
    # # print(multi_df.head())
    # #利用df.query()来取数
    # from_aq_ = "yufa_aq"
    # to_aq_ = "liulihe_aq"
    # str_ =  'from_aq == \"'+from_aq_+'\"and to_aq ==\"'+to_aq_+"\""
    # df = multi_df.query(str_).copy()
    # # print(df['area'].values[0])
    # # print(multi_df.query('from_aq=="yufa_aq"'))
    # df_ = multi_df.query('from_aq=="yufa_aq" and level < 3 and level > 0').copy()
    # print(df_.sort_values(by = 'area'))
    # # aq 最近的meo信息
    # # 整合单个时刻多个数据，计算PM2.5的 AQIs
    # # 输出dataframe 包涵aq_id time pollutions meos(mean) aqis(pm2.5)
    # # day_of_weeek mount
    # aqis = np.zeros(17)
    # #平均PM2.5的浓度，计算分区中PM2.5的AQIs
    # print(aqis)
    # #直接获取某个area值
    # print(multi_df.loc[(from_aq_,to_aq_), 'area'])
    # #
    # print(multi_df.loc[(from_aq_,to_aq_), 'level'])


    ''' 
    # 8 compute aq_aq_neighbor_info
    '''
    aq_aq_df_ = pd.read_csv('../000data_pre_analyze/17_18aq_aq_dis.csv', sep=',', index_col='Unnamed: 0')
    get_aq_aq_neighbors(aq_aq_df_,'17_18aq_aq_neighbor.csv')
    '''
    #9 compute aq_aq_neighbor_info
    '''
    meo_meo_df_ = pd.read_csv('../000data_pre_analyze/17_18meo_meo_dis.csv', sep=',', index_col='Unnamed: 0')
    get_meo_meo_neighbors(meo_meo_df_,'17_18meo_meo_neighbor.csv')




if __name__ == "__main__":
    #计算2017-2018 station信息
    # op_17_18_info()
    #计算2014-2015 station信息
    op_14_15_info()

