import numpy as np
from math import radians, cos, sin, asin, acos, sqrt
import pandas as pd
import torch
import datetime
import numpy as np
from geopy import distance 
def get_AQI_PM2_5(con):  # concentrate -> API
    #input average concentration of PM2.5 one day
    #output AQI of PM2.5
    # Air_pollution_color = ["#00E400","#FFFF00","#FF7E00","#FF0000","#99004C","#7E0023"]
    AQI = 0
    if con>=0 and con<=35:
        #AQI = np.ceil((50-0)/(35-0)*(con-0)+0)
        AQI = np.ceil(10/7*con)
        # color = Air_pollution_color[0]
    elif con>35 and con<=75:
        #AQI = np.ceil((100-50)/(75-35)*(con-35)+50)
        AQI = np.ceil(1.25*(con-35)+50)
        # color = Air_pollution_color[1]
    elif con>75 and con<=115:
        #AQI = np.ceil((150-100)/(115-75)*(con-75)+100)
        AQI = np.ceil(1.25*(con-75)+100)
        # color = Air_pollution_color[2]
    elif con>115 and con<=150:
        #AQI = np.ceil((200-150)/(150-115)*(con-115)+150)
        AQI = np.ceil(10/7*(con-115)+150)
        # color = Air_pollution_color[3]
    elif con>150 and con<=250:
        # AQI = np.ceil((300-200)/(250-150)*(con-150)+200)
        AQI = np.ceil(con+50)
        # color = Air_pollution_color[4]
    elif con>250 and con<=350:
        # AQI = np.ceil((400-300)/(350-250)*(con-250)+300)
        AQI = np.ceil(con+50)
        # color = Air_pollution_color[5]
    elif con>350:
        # AQI = np.ceil((500-400)/(500-350)*(con-350)+400)
        AQI = np.ceil(2/3*(con-350)+400)
    # print(AQI)
    return AQI

def get_AQI(con,type='PM2.5'):
    assert type in ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2'], 'Type is not in pollution types'
    dic = {
        'IAQI' : [0, 50,  100, 150, 200,  300,  400,  500],
        'SO2'  : [0, 150, 500, 650, 800],
        'NO2'  : [0, 100, 200, 700, 1200, 2340, 3090, 3840],
        'CO'   : [0, 5,   10,  35,  60,   90,   120,  150],    # change CO standard
        'PM2.5': [0, 35,  75,  115, 150,  250,  350,  500],
        'PM10' : [0, 50,  150, 250, 350,  420,  500,  600],
        'O3'   : [0, 160, 200, 300, 400,  800,  1000, 1200]
    }
    standard_l = dic[type]
    AQI_l = dic['IAQI']
    # print(standard_l)
    # index = 0
    IAQIp = 0
    for index_ in range(len(standard_l)):
        if standard_l[index_] <= con:
            pass
        else:
            if index_ > 0:
                IAQIp = (AQI_l[index_]-AQI_l[index_-1]) / (standard_l[index_]- standard_l[index_-1])\
                        * (con - standard_l[index_-1]) + AQI_l[index_-1]
                break
            else:
                IAQIp = 0
                break
    if con >= standard_l[index_]:
        IAQIp = AQI_l[index_]
    return int(np.ceil(IAQIp))


def get_AQI_PM2_5_color(aqi): # AQI -> color
    #input aqi of PM2.5
    #output color of PM2.5
    Air_pollution_color = ["#00E400","#FFFF00","#FF7E00","#FF0000","#99004C","#7E0023"]
    if aqi>=0 and aqi<=50:
        color = Air_pollution_color[0]
    elif aqi>50 and aqi<=100:
        color = Air_pollution_color[1]
    elif aqi>100 and aqi<=150:
        color = Air_pollution_color[2]
    elif aqi>150 and aqi<=200:
        color = Air_pollution_color[3]
    elif aqi>200 and aqi<=300:
        color = Air_pollution_color[4]
    elif aqi>300:
        color = Air_pollution_color[5]
    return color
#计算两个地理坐标间的距离

def get_wind_direction_no(angle):
    direction = 0
    if angle > 360:
        direction = 8
    elif angle > 247.5 and angle <= 292.5:
        direction = 0
    elif angle > 292.5 and angle <= 337.5:
        direction = 1
    elif angle > 337.5 or angle <= 22.5:
        direction = 2
    elif angle > 22.5 and angle <= 67.5:
        direction = 3
    elif angle > 67.5 and angle <= 112.5:
        direction = 4
    elif angle > 112.5 and angle <= 157.5:
        direction = 5
    elif angle > 157.5 and angle <= 202.5:
        direction = 6
    elif angle > 202.5 and angle <= 247.5:
        direction = 7
    return direction



def geodistance(lng1,lat1,lng2,lat2):
    #version 1
    # lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    # dlon=lng2-lng1  #弧度差
    # dlat=lat2-lat1
    # a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    # dis=2*asin(sqrt(a))*6371.004*1000
    # return dis
    
    #20200602
    #使用geopy返回 经纬度之间的距离
    #纬度1 经度1 lat1, lng1,
    #纬度2 经度2 lat2, lng2,；
    dis = distance.great_circle((lat1,lng1),(lat2,lng2)).km
    return dis

def get_area(lng1,lat1,lng2,lat2,dis1=30,dis2=100):
    #计算两个经纬度之间的角度 
    # 计算坐标向量与北偏东22.5度的夹角 sita1
    # 两个向量之间的夹角存在二值性，所以用正北单位向量为辅助向量，计算坐标向量与正北向量的夹角 sita2
    # 如果 sita1>sita2 则向量在上方
    # 反之则在下方
    #返回 area[分区，层级]
    # 分区 0-7，
    # level 0-dis1（10km）  dis1-dis2(100km)  >dis2
    #单位向量
    vec1 = np.array([-1* sin(22.5/180* np.pi), cos(22.5/180*np.pi)]) #极坐标系起始位置，北偏东22.5度
    vec1_len = 1
    # print('**2',vec1[0]**2+vec1[1]**2)
    vecy = np.array([0,1])  #
    vecy_len = 1
    #坐标的向量 pos1->pos2
    #斜边
    l = geodistance(lng1, lat1, lng2, lat2)
    if l == 0:     # 0：itself；1 circle1 ； 2:circle2；3:out of area
        level = 0
    elif l <= dis1:
        level = 1
    elif l <= dis2:
        level = 2
    else:
        level = 3
    #水平边
    x = geodistance(lng1, lat1, lng2, lat1)
    #垂直边
    y = geodistance(lng2, lat1, lng2, lat2)
    l = np.sqrt(x**2+y**2)
    if l != 0:
        vec2 = np.array([x, y])
        if lng1 > lng2:
            vec2[0] *= -1
        if lat1 > lat2:
            vec2[1] *= -1
        # print(vec1)
        # print(vec2)
        vet_mul = np.dot(vec1,vec2)   #vec1*vec2 = |vec1||vec2|cos(sita)
        sita = acos(vet_mul/(vec1_len*l))
        # print('sita',sita/np.pi*180)
        vet_assit = np.dot(vecy,vec2)
        sita_assit  = acos(vet_assit/(vecy_len*l))
        # print('sita_assit',sita_assit/np.pi*180)
        if sita_assit < sita: #数据在上方
            area = np.ceil(sita/np.pi*180/45)
        else:
            area = np.ceil((2*np.pi-sita)/np.pi*180/45) 
    else:
        area = 0 #area = 0 表示是自身
    # print('area',area,'level',level)
    return area, level


# def get_random_longiLati(long,lati,minr,maxr,min_sita,max_sita):
#     pass

# def get_one_hot_code(series):
#     one_hot_df = pd.get_dummies(series)
#     # print(ID_one_df)
#     tensor_ = torch.tensor(one_hot_df.values).float().t()
#     # print(tensor_df)
#     # print(tensor_df.shape)
#     return tensor_
def get_M_Week_TimeofDay(date):
    #datatime64-> datatime.datatime
    # time = date.astype(datetime.datetime)
    date = pd.to_datetime(date)
    # time = datetime.datetime.strptime(date,"%Y-%m-%d %H:%M:%S")
    mounth = date.month    #获取日期中的月份
    week = date.weekday()  #获取日期中的星期数
    hour = date.hour       #获取日期中的小时
    # print('mounth',mounth,' ', 'week',week,' ','hour',hour)
    return mounth,week,hour
# def get_SimpleExponentialSmooth(series, bata):
#     # V(t) = sum( V(t-i) * bata * (1-bata)^(t-i) ) / sum( bata * (1-bata)^(t-i) ) = numerator/denominator
#     size = len(series)
#     numerator = 0
#     denominator = 0
#     for i in range(size):
#         temp = bata * (1-bata)**(i)
#         denominator += temp
#         numerator += temp * series[size-1-i]
#     print('Estimator data:',numerator/denominator)
#     return numerator/denominator

# def get_SES_middle(series, bata):  #series is double size
#     size = len(series)
#     assert size % 2 == 0, 'series size must be double'
#     numerator = 0
#     denominator = 0
#     for i in range(size):
#         temp = bata * (1 - bata) ** (i)
#         denominator += temp
#         numerator += temp * series[size - 1 - i]
#     print('Estimator data:', numerator / denominator)
#     return numerator / denominator

def get_IDW_val(dis,arr):
    assert len(dis) == len(arr)
    dis = np.array(dis)
    arr = np.array(arr)
    delete = []
    for i in range(len(dis)):
        if np.isnan(dis[i]) or np.isnan(arr[i]):
            delete.append(i)
    dis = np.delete(dis,delete)
    arr = np.delete(arr,delete)
    if len(dis) != 0:
        dis_sum = 0
        weighted_dis = 0
        for d_ in dis:
            dis_sum += 1/(d_**2)
        # print(dis_sum)
        for i in range(len(dis)):
            weighted_dis += (arr[i]/(dis[i]**2))/ dis_sum
        return weighted_dis
    else:
        return np.nan
# #获取
# def get_meo_inDis(aq_mos_dis,dis):
#     #返回距离内最近的K个临近meo的名称
#     meo_inDis = pd.DataFrame([1])
#     return meo_inDis

# #以监测点为中心合并一个时刻的的气候数据。
# def merge_by_dis(aq_aq_dis,aq_mos_dis,df_aq,df_mos,dis):

#     #获取

#     pass

def get_one_hot_val(val,length):
    #将数值转化为固定长度的one-hot编码
    # print(val)
    # print(length)
    if length == 12:  #月份onehot编码左移动一位, dayofweek 除7取整，timeofday除6取整。包涵0
         val -=1
    assert val < length
    out_nparr = torch.zeros(length,dtype = torch.float)
    # index = torch.long(val)
    val = int(val)
    out_nparr[val] = 1
    # print(out_nparr)
    return out_nparr

def get_one_hot_array(arr,length): #torch.tensor
    #input arr[n*1]
    #output arr[n*m]
    #将数组中的每个值转化为one-hot 编码m*1 转化为m*n
    out_nparr = torch.zeros((len(arr),length), dtype = torch.float) #用于直接计算
    # out_nparr = np.zeros((len(arr),length))   #用于深度学习数据预先训练
    # arr = arr.long()
    for i in range(len(arr)):
        # print(arr[i])
        # print(length)
        val = arr[i]
        if length == 12:  #月份onehot编码左移动一位, dayofweek 除7取整，timeofday除6取整。包涵0
            val -= 1
        assert val < length
        out_nparr[i,(int)(val)] = 1
    # print(out_nparr)
    return out_nparr

def cat_onehot_df(tensor_arr,length = [12,7,4]):
    # length = [月份12,星期7,一天时刻4]
    # result = np.array([])
    #将dataframe格式的数据中的行逐个数值转化为one-hot编码，并且拼接
    first = True
    for line in range(len(tensor_arr)):
        arr =  tensor_arr[line,:]
        for i  in range(len(arr)):
            if i == 0:
                val = get_one_hot_val(arr[i],length[i])
                temp = np.array(val)  
            else:
                # temp.extend(get_one_hot_val(arr[i],length[i]))
                temp = np.hstack((temp, get_one_hot_val(arr[i],length[i])))
        if first:
            first = False
            result = temp 
        else:
            result = np.vstack((result,temp))
    result = torch.from_numpy(result)
    return result

def get_evaluation_index(output_matrix, origin_matrix):
    #input data: numpy.nparr
    assert output_matrix.shape == origin_matrix.shape, \
        'The size of Two matrixes are not match, please check the code'
    rows, columns_size = output_matrix.shape
    #
    # absolute_error_sum = np.zeros(columns_size)
    # sum_original_data_sum = np.zeros(columns_size)
    # print('matrix size:', rows, columns_size)
    #for each step
    error_matrix = output_matrix - origin_matrix
    # print('error_matrix', error_matrix)
    abs_matrix = abs(error_matrix)
    # print('abs_matrix', abs_matrix)
    abs_originmatrix = abs(origin_matrix)
    # print('abs_originmatrix',abs_originmatrix)
    absolute_error_sum_foreachcolumns = np.sum(abs_matrix, axis=0)
    # print('absolute_error_sum_foreachcolumns',absolute_error_sum_foreachcolumns)
    original_data_sum_foreachcolumns = np.sum(abs_originmatrix, axis=0)
    # print('original_data_sum_foreachcolumns',original_data_sum_foreachcolumns)
    acc_foreachcolumns = 1 - absolute_error_sum_foreachcolumns/original_data_sum_foreachcolumns
    mae_foreachcolumns = absolute_error_sum_foreachcolumns / rows
    # print('acc', acc_foreachcolumns)
    # print('mae', mae_foreachcolumns)


    #for all data
    absolute_error_sum = np.sum(absolute_error_sum_foreachcolumns)
    print('absolute_error_sum',absolute_error_sum)
    original_data_sum = np.sum(original_data_sum_foreachcolumns)
    print('original_data_sum',original_data_sum)
    acc_all = 1 - absolute_error_sum/original_data_sum
    print('acc_all',acc_all)
    mae_all = absolute_error_sum/(rows * columns_size)
    print('mae_all',mae_all)
    return acc_foreachcolumns, mae_foreachcolumns, acc_all, mae_all


import time
def form_debug_filename(epoch,batch,lr,fold,normtype):
    # time_int = time.time()
    local_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    # print(time.time())
    # print(local_time)
    trainlogName = 'logs/debug/'+local_time+'_epoch='+str(epoch)+'_batch='+str(batch)+'_lr='+str(lr)+'_fold='+str(fold)+'_'+normtype+'train.txt'
    testlogName = 'logs/debug/'+local_time+'_epoch='+str(epoch)+'_batch='+str(batch)+'_lr='+str(lr)+'_fold='+str(fold)+'_'+normtype+'test.txt'

    return trainlogName,testlogName

#20200812
def getMAE(realY,estiY):
    # get mean absolute error
    assert len(realY) == len(estiY), "file size is not equal"
    sum = 0.0
    num = len(realY)
    count = 0
    # for y1,y2 in realY, estiY:
    for i in range(len(realY)):
        if np.isnan(estiY[i]) or np.isnan(realY[i]):
            print(i, ' getMAE exit NaN')
            continue
        else:
            sum += abs(realY[i]-estiY[i])
            count+=1
    return sum/count

def getMRE(realY,estiY):
    assert len(realY) == len(estiY), "file size is not equal"
    numerator = 0.0
    denominator = 0.0
    # for y1, y2 in realY, estiY:
    for i in range(len(realY)):
        if np.isnan(estiY[i]) or np.isnan(realY[i]):
            pass
        else:
            numerator += abs(realY[i] - estiY[i])
            denominator += abs(realY[i])
    return numerator / denominator

def compute_score_byInterval(df_,item,method,datalength,interval=1000):
        #TODO 本函数用于计算各个时间段内目标插入值的性能，以interval为样本点间隔
        # 比较 self.test_index 对应值与self.originaldata中对应数值之间的mse,
        #填充估计值
        for row in df_.index:
            if row%5000==0:
                print('Process')

        df_.to_csv( method + item+'test_score.csv')
        mae = []
        mre = []
        subdatasize = []
        size = int(np.ceil(datalength/interval))#TODO 定义数据的区段数
        for i in range(size):
            start_index = i*interval
            end_index = (i+1)*interval
            if end_index >= datalength:
                end_index = datalength-1
            subdf_ = df_[(df_['row']>=start_index) & (df_['row']< end_index)].copy()
            reald = subdf_['data'].values
            estid = subdf_['esti_data'].values
            mae.append(getMAE(reald, estid))
            mre.append(getMRE(reald, estid))
            subdatasize.append(len(subdf_))
        reald_all = df_['data'].values
        estid_all = df_['esti_data'].values
        mae_all = getMAE(reald_all, estid_all)
        mre_all = getMRE(reald_all, estid_all)
        with open('result/output_result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
            # print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
            for i in range(size):
                print(method,'Range from',i*interval,'to',(i+1)*interval, 'datasize:',subdatasize[i],'MAE', mae[i], 'MRE', mre[i], file=fileHandle)
            print(method,'Range from begin to end','MAE', mae_all, 'MRE', mre_all, file=fileHandle)
        return mae, mre,mae_all,mre_all
def _diff(arr):
    diff = np.zeros(len(arr))
    for i in range(1,len(arr)):
        if not np.isnan(arr[i]-arr[i-1]):
            diff[i] = arr[i]-arr[i-1]
    return diff

def weighted_sum(arr,weight):
    arr = np.array(arr).reshape(-1)
    weight = np.array(weight).reshape(-1)
    if not np.isnan(arr).any(): #arr 中不存在异常
        pass
    else:
        pass
    val_sum = 0
    weight_sum = 0
    for i in range(len(arr)):
        if np.isnan(arr[i]) or np.isnan(weight[i] ):
            pass
        else:
            val_sum += weight[i] * arr[i]
            weight_sum += weight[i]
    if weight_sum == 0:
        return np.nan
    else:
        return val_sum/weight_sum
        #return val_sum
if __name__ == "__main__":
    PM2_5_cons = [0,20,35,60,75,100,115,130,150,200,250,300,350,400,500]
    for con in PM2_5_cons:
        aqi = get_AQI_PM2_5(con)
        print(aqi)
        color = get_AQI_PM2_5_color(aqi)
        print(color)
    a = [1,2,3,4,5]
    # print(a[:2])

    ##test compute distance 
    dis1 = geodistance(116,0,117,0)
    print('赤道经度差一度的距离m',dis1)
    '''
    # test aq_aq_dis

    '''
    dis1 = geodistance(116.3,39.52,116,39.58)
    print('yufa_aq to liulihe_aq',dis1)
    #beibuxinqu_aq	40.09	116.174
    dis1 = geodistance(116.3,39.52,116.174,40.09)
    print('yufa_aq to beibuxinqu_aq',dis1)
    dis1 = geodistance(116.663,39.886,116.106,39.937)
    print('tongzhou_aq to mentougou_aq',dis1)

    '''
    # test aq_meo_dis
    '''
    dis = geodistance(0,1,0,2)
    print('test geopy',dis)
    dis1 = geodistance(116.3,39.52,115.9688889,40.44944444)
    print('yufa_aq to yanqing_meo',dis1)

    dis1 = geodistance(116.483,39.939,116.2052778,39.9425)
    print('dongsihuan_aq to shijingshan_meo',dis1)

    dis1 = geodistance(116.628,40.328,116.5008333,39.9525)
    print('huairou_aq to chaoyang_meo',dis1)


    get_M_Week_TimeofDay('2020-05-30 10:00:00')

    dis1 = geodistance(115.45536883,39.64535368,116.417,39.929)
    print(dis1)  #88.01169859882543

    dis1 = geodistance(115.45536883,39.64535368,116.48299999999999 ,39.939)
    print(dis1)  #93.67407230552429


    print(get_IDW_val([2,5,1,4,1],[1,2,3,4,np.nan]))
    print(get_IDW_val([2,5,1,4,np.nan],[1,2,3,4,1]))

    print('test get_one_hot_array')
    arr = [0,1,2,3,5,6,7,8] 
    onehot_arr = get_one_hot_array(arr,10)


    #test embedding
    # import torch
    # input =  torch.from_numpy(onehot_arr).double()
    # print(input.size())
    # w = torch.rand(10, 3).double()
    # print(w.size())
    # print(w)
    # out = input.mm(w)
    # print(out)


    # test get_SimpleExponentialSmooth(series, bata):
    # series = [1,2,3,4,5,6,7,8,9,10]
    # bata = 0.8
    # val = get_SimpleExponentialSmooth(series, bata)
    # print(val)
    print('')
    import numpy as np
    print('testing computing evaluation_index')
    out = np.random.randn(1, 6)
    origin = np.random.randn(1, 6)
    acc_foreachcloumns, mae_foreachcloumns, acc, mae\
        = get_evaluation_index(out, origin)
    print(acc_foreachcloumns)
    print(mae_foreachcloumns)
    print(acc)
    print(mae)

    '''
    'IAQI' : [0, 50,  100, 150, 200,  300,  400,  500],
    'SO2'  : [0, 150, 500, 650, 800],
    'NO2'  : [0, 100, 200, 700, 1200, 2340, 3090, 3840],
    'CO'   : [0, 2,   4,   14,  24,   36,   48,   60],
    'PM2_5': [0, 35,  75,  115, 150,  250,  350,  500],
    'PM10' : [0, 50,  150, 250, 350,  420,  500,  600],
    'O3'   : [0, 160, 200, 300, 400,  800,  1000, 1200]
    '''
    for pm25 in [0,20,49,50,100,150,180,250,300,360,500,510,45,49]:
        print('pm2.5 con',pm25)
        print('get_AQIP2_5',get_AQI_PM2_5(pm25))
        print('get_AQI',get_AQI(pm25,'PM2.5'))
    # aqi = get_AQI(600, 'PM2_5')
    # print(int(aqi))
    # aqi = get_AQI(700, 'PM10')
    # print(aqi)
    # aqi = get_AQI(900, 'SO2')
    # print(aqi)
    # aqi = get_AQI(2000, 'NO2')
    # print(aqi)
    # aqi = get_AQI(1000, 'O3')
    # print(aqi)
    # aqi = get_AQI(160, 'CO')
    # print(aqi)

    # aqi = get_AQI(600,'O3')
    # print(aqi)
    print('get_AQI', get_AQI(93.755, 'PM10'))
    print('get_AQI', get_AQI(69, 'NO2'))
    print('get_AQI', get_AQI(2.4, 'CO'))
    print('get_AQI', get_AQI(15, 'O3'))
    print('get_AQI', get_AQI(34, 'SO2'))


    for windir in [250,290,310,25,180,190]:
        print(get_wind_direction_no(windir))

    get_evaluation_index(np.array([[2,4,6,8,10]]),np.array([[1,2,3,4,5]]))

    # test diff function
    arr = [120,200,np.nan,np.nan,np.nan]
    print(arr)
    print(_diff(arr))