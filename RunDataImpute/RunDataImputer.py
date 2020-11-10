
from ImputerOp.IMA import *

warnings.filterwarnings("ignore")
#TODO sklearn 模型存储与加载
import joblib
# 保存模型 joblib.dump(prams,'prams.pkl')
# 加载模型 ＃prams = joblib.load("prams.pkl")
def merge_two_datasets(df1,df2,by_):
    new_df = pd.concat([df1,df2])
    if 'utc_time' in new_df.columns:  # for those time series data
        new_df['utc_time'] = pd.to_datetime(new_df['utc_time'])
    new_df = new_df.sort_values(by=by_,ascending=[True,True])
    return  new_df


def run_imputing_aq(subitem, year, outfile):
    #subitem用于寻找K近邻。
    if year == '14_15':
        data_path = "../000data_pre_analyze/beijing20142015/beijing14_15_aq.csv"
        dis_file = "../000data_pre_analyze/beijing20142015/beijing14_15aq_dis.csv"
        pearson_neighbor_file = 'origin/'+year + subitem + 'pearson_neighbor.csv'
        # pearson_neighbor_dis_file = 'origin/'+year + item + 'pearson_neighbor_dis.csv'
        pearson_neighbor_dis_file = '../000data_pre_analyze/beijing20142015/14_15aq_aq_dis.csv'
    elif year == '17_18':
        data_path = "../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv"
        dis_file = "../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv"
        pearson_neighbor_file ='origin/'+ year + subitem + 'pearson_neighbor.csv'
        # pearson_neighbor_dis_file ='origin/' + year + item + 'pearson_neighbor_dis.csv'
        pearson_neighbor_dis_file ='../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv'
    elif year =='1802_1803':
        data_path = "../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv"

        dis_file = "../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv"
        pearson_neighbor_file = 'origin/' + year + subitem + 'pearson_neighbor.csv'
        # pearson_neighbor_dis_file ='origin/' + year + item + 'pearson_neighbor_dis.csv'
        pearson_neighbor_dis_file = '../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv'
    else:
        assert 'Input right year: 14_15 or 17_18'

    origindata = pd.read_csv(data_path)
    #TODO 计算皮尔森相关系数领域
    temp_path = Path(pearson_neighbor_file)
    temp_path2 = Path(pearson_neighbor_dis_file)
    if temp_path.exists() and temp_path2.exists():
        pass
    else: #计算

        dis_df = pd.read_csv(dis_file, index_col='Unnamed: 0')
        neighbor_info, neighbor_dis_info = get_pearson_neighbor(origindata,dis_df,subitem,year)



    neighbor_info = pd.read_csv(pearson_neighbor_file,  # 邻居信息
                                index_col='Unnamed: 0')
    neighbor_info = neighbor_info.sort_index()  # 重新排序index
    neighbor_dis_info = pd.read_csv(pearson_neighbor_dis_file,  # 邻居距离信息
                                    index_col='Unnamed: 0')
    item = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    check_list= pd.DataFrame()
    #item1:插值参考属性集合， item2表示待插值的属性，两者相同，表示使用对全部属性进行，插值
    ima = IMA(origindata, check_list, item, item,'auxiliaryFile/'+outfile+'/', neighbor_info,
                  neighbor_dis_info, knnVal=7, wind=11)
    ima.run()
    finalData = ima.originaldata.get_mergedf()
    columns = ['station_id','utc_time','PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    finalData = finalData[columns]
    finalData.to_csv('../FinalDatasetsForDNN/'+year+'aq.csv',index=False)

def run_imputing_aq_oneModel(subitem, year, outfile):
    #subitem用于寻找K近邻。


    data17_18_path = "../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv"
    dis_file = "../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv"
    pearson_neighbor_file ='origin/'+ year + subitem + 'pearson_neighbor.csv'
    pearson_neighbor_dis_file ='../000data_pre_analyze/beijing20172018/17_18aq_aq_dis.csv'
    data180203_path = "../000data_pre_analyze/beijing20172018/beijing201802_201803_aq.csv"

    origindata1 = pd.read_csv(data17_18_path)

    origindata2 = pd.read_csv(data180203_path)
    fill_nan_df = pd.DataFrame(columns=origindata2.columns)
    size = len(origindata2[origindata2['station_id'] == 'aotizhongxin_aq'])
    #由于20180203缺失zhiwuyuan的数据，添加nan值补全
    fill_nan_df['station_id'] = ['zhiwuyuan_aq' for i in range(size)]
    fill_nan_df['utc_time'] = origindata2[origindata2['station_id'] == 'aotizhongxin_aq']['utc_time']
    fill_nan_df['Unnamed: 0'] = origindata2[origindata2['station_id'] == 'aotizhongxin_aq']['Unnamed: 0']
    fill_nan_df['Unnamed: 0.1'] = origindata2[origindata2['station_id'] == 'aotizhongxin_aq']['Unnamed: 0.1']

    origindata2 = pd.concat([origindata2,fill_nan_df])
    origindata2['Unnamed: 0']  = origindata2['Unnamed: 0'] + 9482
    origindata2['Unnamed: 0.1'] = origindata2['Unnamed: 0.1'] + 9482
    origindata = merge_two_datasets(origindata1,origindata2, by_=['station_id','utc_time'])
    #TODO 计算皮尔森相关系数领域
    temp_path = Path(pearson_neighbor_file)
    temp_path2 = Path(pearson_neighbor_dis_file)
    if temp_path.exists() and temp_path2.exists():
        pass
    else: #计算

        dis_df = pd.read_csv(dis_file, index_col='Unnamed: 0')
        neighbor_info, neighbor_dis_info = get_pearson_neighbor(origindata,dis_df,subitem,year)



    neighbor_info = pd.read_csv(pearson_neighbor_file,  # 邻居信息
                                index_col='Unnamed: 0')
    neighbor_info = neighbor_info.sort_index()  # 重新排序index
    neighbor_dis_info = pd.read_csv(pearson_neighbor_dis_file,  # 邻居距离信息
                                    index_col='Unnamed: 0')
    item = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    check_list= pd.DataFrame()
    #item1:插值参考属性集合， item2表示待插值的属性，两者相同，表示使用对全部属性进行，插值
    ima = IMA(origindata, check_list, item, item,'auxiliaryFile/'+outfile+'/', neighbor_info,
                  neighbor_dis_info, knnVal=7, wind=11)
    ima.run()
    finalData = ima.originaldata.get_mergedf()
    columns = ['station_id','utc_time','PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    finalData = finalData[columns]
    finalData.to_csv('../FinalDatasetsForDNN/'+year+'aq.csv',index=False)


if __name__ == '__main__':
    #TODO 此处不需要验证插值性能 run的参数为文件夹命名.
    # 运行单个数据集
    #17_18年的数据插值
    # run_imputing_aq('PM2.5', '17_18', '17_18')
    #18 02-03的数据插值
    # run_imputing_aq('PM2.5', '1802_1803', '1802_1803')

    #TODO 整合20172018和20180203数据集，进行插值操作
    run_imputing_aq_oneModel('PM2.5','17_18','20172018')