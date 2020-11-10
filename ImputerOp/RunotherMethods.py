from ImputerOp.compareImputingMethods import *
from pathlib import Path
def run_test(item,year,outfile,interval):
    with open('result/output_result.txt', mode='a') as fileHandle:  # 'w'写，'r'读，'a'添加
        # print(str_, ' ', 'MAE:', mae, ' ', 'MRE:', mre, file=fileHandle)
        print('Running test for ', year,' ',item, file=fileHandle)
    #TODO outfile demo: 17_18_PM2_5
    if year == '14_15':
        data_path = "../000data_pre_analyze/beijing20142015/beijing14_15_aq.csv"
        loss_log_output_path = 'origin/checkloss14_15_aq'+item+'.csv'

    elif year=='17_18':
        data_path = "../000data_pre_analyze/beijing20172018/beijing17_18_aq.csv"
        loss_log_output_path = 'origin/checkloss17_18_aq'+item+'.csv'
    elif year=='1802_1803':
        data_path = "../000data_pre_analyze/beijing20172018/beijing201802_201803_aq.csv"
        loss_log_output_path = 'origin/checkloss1802_1803_aq' + item + '.csv'
    else:
        assert 'Input right year: 14_15 or 17_18'
    #TODO 生成文件缺失记录文件
    temp_path = Path(loss_log_output_path)
    if not temp_path.exists():  #TODO run once 如果文件存在则跳过
        data = pd.read_csv(data_path)
        check_loss(data, item, loss_log_output_path) #核对缺失数据，并存储在输出文件中

    #TODO 删除特定位置的数据，用于插值验证 #PM2.5在文件命名中存在问题
    str = outfile + '.csv'
    del_df_file = "origin/del_testdata_" + str
    temp_path = Path(del_df_file)
    if not temp_path.exists():  #TODO run once 如果文件存在则跳过
    # if True:  #TODO run once 如果文件存在则跳过
        check_list = pd.read_csv(loss_log_output_path)

        data = pd.read_csv(data_path)
        del_data_by_index(data, check_list, item, str)



    # print('1:Running Global mean inputing method')
    # check_list = pd.read_csv(loss_log_output_path)
    # data = pd.read_csv(data_path)
    # #oped_data = pd.read_csv("../111CompareImputer/del_testdata_17_18_PM2_5.csv")
    # insert_global_mean(data,check_list,item,str,interval)
    # # print(mae,mre)
    #
    # print('2:Running Local mean inputing method')
    # check_list = pd.read_csv(loss_log_output_path)
    # data = pd.read_csv(data_path)
    # path = "origin/del_testdata_"+str
    # oped_data = pd.read_csv(path)
    # w = 13
    # insert_history_mean(data,oped_data,check_list,item,w,str,interval)
    # # print(mae, mre)
    #
    # print('3:Running xgboost')
    # check_list = pd.read_csv(loss_log_output_path)
    # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
    # data = pd.read_csv(data_path)
    # path = "origin/del_testdata_" + str
    # oped_data = pd.read_csv(path)
    # insert_data_bymethod(data, oped_data, check_list, item, str,'xgboost',interval)
    # # print(mae, mre)
    #
    # print('4:Running lightgbm')
    # check_list = pd.read_csv(loss_log_output_path)
    # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
    # data = pd.read_csv(data_path)
    # path = "origin/del_testdata_" + str
    # oped_data = pd.read_csv(path)
    # insert_data_bymethod(data, oped_data, check_list, item, str, 'lightgbm',interval)
    # # print(mae, mre)
    #
    # print('5:Running knn')
    # check_list = pd.read_csv(loss_log_output_path)
    # # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
    # data = pd.read_csv(data_path)
    # path = "origin/del_testdata_" + str
    # oped_data = pd.read_csv(path)
    # insert_data_bymethod(data, oped_data, check_list, item, str, 'knn',interval)
    # # print(mae, mre)

    #很慢
    print('6:Running randomforest')
    check_list = pd.read_csv(loss_log_output_path)
    # check_list = check_list[(check_list['station_id'] =='aotizhongxin_aq') | (check_list['station_id'] =='badaling_aq')]
    data = pd.read_csv(data_path)
    path = "origin/del_testdata_" + str
    oped_data = pd.read_csv(path)
    insert_data_bymethod(data, oped_data, check_list, item, str, 'randomforest',interval)
    # print(mae, mre)



if __name__ == '__main__':
    #TODO 测试17_18的插值效果
    # run_test('NO2', '17_18', '17_18_NO2',2000)
    # run_test('PM2.5', '17_18', '17_18',2000)

    #TODO 测试1802-1803的插值效果
    # run_test('NO2', '1802_1803', '1802_1803',1000)
    # run_test('PM2.5', '1802_1803', '1802_1803_PM2.5',1000)


    #TODO 测试14_15的插值效果
    # run_test('PM2.5', '14_15', '14_15_PM2.5',2000)
    run_test('NO2','14_15','14_15_NO2',2000)