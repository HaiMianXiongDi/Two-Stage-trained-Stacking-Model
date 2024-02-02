from XGBoostFun import XGBoost_fun
from SVRFun import SVR_fun
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import math
import os


class Stacking_fun():


    def __init__(self):
        self.XGB_item = XGBoost_fun()
        self.SVR_item = SVR_fun()
        self.window_size_rate = 0.1
        # 参数配置
        'Temp'
        'Exchange'
        'Electricity'
        self.csv_file_name = 'Exchange'
        self.csv_file_path = 'Datasets/' + self.csv_file_name + '.csv'  # CSV文件路径
        self.file_column_index = 1 #文件里的value列：0,1,2,3，...

        self.XGB_item.csv_file_name = self.csv_file_name
        self.XGB_item.XGBoost_window_size_rate=self.window_size_rate
        
        self.SVR_item.csv_file_name = self.csv_file_name
        self.SVR_item.SVR_window_size_rate=self.window_size_rate

    def load_data(self, csv_file_path):
            try:
                data = pd.read_csv(csv_file_path)
                if data is not None and self.file_column_index < len(data.columns):
                    value_list = data.iloc[:, self.file_column_index].tolist()
                    # 检查每个元素，如果它不是浮点数，则尝试转换
                    value_list = [float(item) if not isinstance(item, float) else item for item in value_list]
                    return value_list
                else:
                    print("数据列错误或文件为空")


            except pd.errors.ParserError as e:
                print(f"Error reading CSV file: {e}")
            return None

    def preprocess_mimmax(self, value_list):
            # 归一化
            value_min = np.min(value_list)
            value_max = np.max(value_list)
            value_list = ((value_list - value_min) / (value_max - value_min)).astype(np.float32)

            return value_list
   
    def canva2_combined_update(self, y_actual, y_pred_on_actual, y_future_pred, window_size, split_index=0.8, title='Stacking Plot of '):

        title = title + self.csv_file_name
        # 绘图设置
        plt.figure(figsize=(14, 8))
        plt.title(title, fontsize=20)
        plt.xlabel("Timestamp", fontsize=20)
        plt.ylabel("Value", fontsize=20)
        plt.tick_params(labelsize=10)
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.5)

        # 绘制数据库中获取的数据
        #plt.plot(x, y, label='Database Data', color='Green', linewidth=1.5)

        # 以下是原 canva2_prdict_update 中的代码，稍作修改
        y_future_pred = np.squeeze(y_future_pred)
        split_point = int(split_index * len(y_actual))
        
        plt.plot(y_actual[:split_point], label='Original_data:80%', color='RoyalBlue', linewidth=1)
        plt.plot(range(split_point, len(y_actual)), y_actual[split_point:], label='Original_data 20%', color='LightSkyBlue', linewidth=1)

        if y_pred_on_actual is not None:
            plt.plot(range(window_size, window_size + len(y_pred_on_actual)), y_pred_on_actual, label='Predicted on Training Set', color='red', linewidth=1.5)

        plt.plot(range(split_point, split_point + len(y_future_pred)), y_future_pred, color='Magenta', linewidth=1.5, label='Predicted on Test Set')
        plt.axvline(x=split_point, color='gray', linestyle='--', linewidth=1.5)

       # plt.legend(fontsize=18, loc='upper left')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08)

        # 保存图像到指定路径
        save_path = f'Images/{self.csv_file_name}-Stacking'
        plt.savefig(save_path)  # 保存图像
        #plt.show()

    def create_lagged_features(self, data, window_size):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i])
            y.append(data[i])
        X, y = np.array(X), np.array(y)
        return X, y

    def prepare_data(self, Source_data):
        value_list = Source_data

        index_60_percent = int(len(value_list) * 0.6)
        index_80_percent = int(len(value_list) * 0.8)

        # 获取前60%的数据作为 train_data
        train_data = value_list[:index_60_percent]
        # 从60% ~ 80%作为validation_data
        validation_data = value_list[index_60_percent:index_80_percent]

        # 将时间序列数据转换为监督学习数据
        X_train, y_train = self.create_lagged_features(train_data, self.window_size)
        X_val, y_val = self.create_lagged_features(validation_data, self.window_size)

        return X_train, y_train, X_val, y_val

    def Stacking_run_Predciting(self, value_list):
        
        value_list = self.preprocess_mimmax(value_list)
        self.window_size = int(self.window_size_rate * len(value_list))
  

        # 计算80%的位置
        value_80 = len(value_list) * 0.8
        value_60 = len(value_list) * 0.6

        split_index_80 = math.ceil(len(value_list) * 0.8)
        split_index_60 = math.ceil(len(value_list) * 0.6)

        is_decimal_80 = value_80 != math.floor(value_80)
        is_decimal_60 = value_60 != math.floor(value_60)


        # 获取前80%的数据
        value_list_80 = value_list[:split_index_80]
        # 获取前60%的数据
        value_list_60 = value_list[:split_index_60]
        # 获取后20%的数据（即80%到100%的数据）
        value_list_20 = value_list[split_index_80:]
        # 从self.window_size开始获取数据
        if(is_decimal_60):
            value_list_60_from_window = value_list_60[self.window_size+1:]
        else:
            value_list_60_from_window = value_list_60[self.window_size:]


        # 为了获取value_list_80中的后20%的数据，我们需要在value_list_80中计算出相应的索引
        split_index_80_in_80 = math.ceil(len(value_list_80) * 0.8)
        # 获取value_list_80中后20%的数据（即原始列表的64%到80%的部分）
        value_list_80_last_20 = value_list_80[split_index_80_in_80:]



        
        xgboost_y_pred_train , xgboost_future_predictions = self.XGB_item.XGBoost_run_TrainPredcit_test(value_list, 0)
        svr_y_pred_train , svr_future_predictions = self.SVR_item.SVR_run_TrainPredcit_test(value_list, 0)



        # 找出长度最短的数组长度
        min_length = min(len(xgboost_future_predictions), len(svr_future_predictions), len(value_list_80_last_20))

        # 截取xgboost和svr的预测结果以及value_list_80_last_20，保证长度一致
        xgboost_future_predictions_aligned = xgboost_future_predictions[:min_length]
        svr_future_predictions_aligned = svr_future_predictions[:min_length]
        value_list_80_last_20_aligned = value_list_80_last_20[:min_length]

        # 现在可以安全地进行堆叠和训练了
        stacked_features_future = np.column_stack((xgboost_future_predictions_aligned, svr_future_predictions_aligned))
        meta_model = LinearRegression()
        meta_model.fit(stacked_features_future, value_list_80_last_20_aligned)


        '''
        print("长度（value_list_80）:", len(value_list_80))
        print("长度（value_list_20）:", len(value_list_20))
        print("长度（split_index_80_in_80）:", split_index_80_in_80)
        print("长度（value_list_80_last_20）:", len(value_list_80_last_20))

        print("长度（xgboost_y_pred_train）:", len(xgboost_y_pred_train))
        print("长度（xgboost_future_predictions）:", len(xgboost_future_predictions))
        print("长度（svr_y_pred_train）:", len(svr_y_pred_train))
        print("长度（svr_future_predictions）:", len(svr_future_predictions))
        '''


        # ------------------预测未来数据---------------------------#
        xgboost_y_pred_train , xgboost_future_predictions = self.XGB_item.XGBoost_run_TrainPredcit(value_list, 1)
        svr_y_pred_train , svr_future_predictions = self.SVR_item.SVR_run_TrainPredcit(value_list, 1)

        stacked_features_train = np.column_stack((xgboost_y_pred_train, svr_y_pred_train))
        stacked_features_future = np.column_stack((xgboost_future_predictions, svr_future_predictions))


        stacked_train_predictions = meta_model.predict(stacked_features_train)
        stacked_future_predictions = meta_model.predict(stacked_features_future)

        #self.canva2_combined_update(value_list,stacked_train_predictions,stacked_future_predictions,self.XGB_item.XGBoost_window_size)

        mse_train = np.mean((stacked_train_predictions - value_list_60_from_window ) ** 2)
        mae_train = np.mean(np.abs(stacked_train_predictions - value_list_60_from_window ))	
        mse_test = np.mean((stacked_future_predictions - value_list_20) ** 2)
        mae_test = np.mean(np.abs(stacked_future_predictions - value_list_20))


        def save_stk_results(mae_train, mse_train, mae_test, mse_test):
            # 确保EXP-Details文件夹存在
            if not os.path.exists('EXP-Details'):
                os.makedirs('EXP-Details')

            # 定义文件路径
            file_path = f'EXP-Details/{self.csv_file_name}-Stacking-mae+mse.txt'

            # 打开文件进行写入
            with open(file_path, 'w') as file:
                file.write("="*25 + f" Stacking Results of {self.csv_file_name} " + "="*25 + "\n")
                file.write(f"MAE Train: {mae_train}\n")
                file.write(f"MSE Train: {mse_train}\n")
                file.write("-"*50 + "\n")
                file.write(f"MAE Future: {mae_test}\n")
                file.write(f"MSE Future: {mse_test}\n")
                file.write("="*50 + "\n")
                file.write("\n\n")
        save_stk_results(mae_train, mse_train, mae_test, mse_test)

        '''
        save_xgb_results(mae_train, mse_train, mae_test, mse_test)
        print("="*25 + f" Stacking Results of {self.csv_file_name} " + "="*25)
        print(f"MAE Train: {mae_train}")
        print(f"MSE Train: {mse_train}")
        print("-"*50)
        print(f"MAE Future: {mae_test}")
        print(f"MSE Future: {mse_test}")
        self.mse_xgboost=mse_test
        print("="*50)
        print("")
        print("")
        '''
        self.canva2_combined_update(value_list,stacked_train_predictions,stacked_future_predictions,self.XGB_item.XGBoost_window_size)

        print("Experiment all finished! Check the results in /EXP-Details and /Images")
        return


if __name__ == "__main__":
    STK_item = Stacking_fun()
    value_list = STK_item.load_data(STK_item.csv_file_path)
    STK_item.Stacking_run_Predciting(value_list)

    