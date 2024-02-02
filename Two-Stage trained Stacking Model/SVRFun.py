
# 

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math

class SVR_fun():
#---------------------------------------------------------------------#

    def __init__(self):

        self.SVR_train_size = 0.8# 训练集+验证集大小
        self.num_boost_round = 100
        self.SVR_window_size_rate = 0.1  #原始数据的长度 * SVR_window_size_rate
        self.SVR_predict_rate = 0.2

        # 参数配置
        self.kernel = 'rbf'
        self.C = 1.5
        self.epsilon = 0.1
        # 参数配置
        'Temp'
        'Exchange'
        'Electricity'
        self.csv_file_name = 'Temp'
        self.csv_file_path = 'Datasets/' + self.csv_file_name + '.csv'  # CSV文件路径
        self.file_column_index = 1 #文件里的value列：0,1,2,3，...

    def SVR_create_lagged_features(self, data, SVR_window_size):
        X, y = [], []
        for i in range(SVR_window_size, len(data)):
            X.append(data[i-SVR_window_size:i])
            y.append(data[i])
        X, y = np.array(X), np.array(y)
        return X, y

    def SVR_prepare_data(self, Source_data):
        value_list = Source_data

        index_60_percent = int(len(value_list) * 0.6)
        index_80_percent = int(len(value_list) * 0.8)

        # 获取前60%的数据作为 train_data
        train_data = value_list[:index_60_percent]
        # 从60% ~ 80%作为validation_data
        validation_data = value_list[index_60_percent:index_80_percent]

        # 将时间序列数据转换为监督学习数据
        X_train, y_train = self.SVR_create_lagged_features(train_data, self.SVR_window_size)
        X_val, y_val = self.SVR_create_lagged_features(validation_data, self.SVR_window_size)

        return X_train, y_train, X_val, y_val

    def SVR_model_train(self, x_train, y_train, x_val, y_val):
        # 创建SVR模型实例    
        model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, verbose=True)
        model.fit(x_train, y_train)        
        # 返回训练好的模型
        return model

    def SVR_model_predict(self, svm_model, X_data):
        predictions = svm_model.predict(X_data)
        return predictions
    
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

    def SVR_run_TrainPredcit(self, value_list, if_polt):
        
        self.SVR_window_size = int(self.SVR_window_size_rate * len(value_list))



        #--------------------检查窗口大小与数据量的关系-------------------# 
        if self.SVR_window_size_rate >= 0.6 :
            print(f'参数错误：模型窗口({self.SVR_window_size})大于训练集')
            return
        if self.SVR_window_size_rate >= 0.2:
            print(f'参数错误：模型窗口({self.SVR_window_size})大于验证集')
            return
        #--------------------检查窗口大小与数据量的关系-------------------# 
        


        #--------------------数据归一化-------------------# 
        value_list = self.preprocess_mimmax(value_list)
        #--------------------数据归一化-------------------# 



        #--------------------模型训练-------------------#
        X_train, y_train, X_val, y_val = self.SVR_prepare_data(value_list)
        xgb_model = self.SVR_model_train(X_train, y_train, X_val, y_val)
        #--------------------模型训练-------------------#



        print("Predicting... (｀・ω・´)")





        #--------------------预测训练集上数据-------------------#
        y_pred_train = []
        for window in X_train:
            prediction = self.SVR_model_predict(xgb_model, window.reshape(1, -1))
            y_pred_train.append(prediction[0])
        y_pred_train = np.array(y_pred_train)
        #--------------------预测训练集上数据-------------------#



        #--------------------计算训练集上的MSE和MAE-------------------#
        mse_train = np.mean((y_pred_train - y_train) ** 2)
        mae_train = np.mean(np.abs(y_pred_train - y_train))
        #--------------------计算训练集上的MSE和MAE-------------------#


        #--------------------分割训练集和测试集 8/10  2/10-------------------#
        x_list_train, x_list_test = train_test_split(value_list, test_size=0.20, shuffle=False)
        #--------------------分割训练集和测试集 8/10  2/10-------------------#




        # 预测未来 n 步
        self.future_steps = int(self.SVR_predict_rate * len(value_list))
        if self.future_steps > 0:
            last_window = x_list_train[-self.SVR_window_size:].tolist()  # 使用训练数据的最后一个窗口
            future_predictions = []
            steps_to_predict = self.future_steps
            for _ in range(steps_to_predict):
                window = np.array(last_window[-self.SVR_window_size:]).reshape(1, -1)
                prediction = self.SVR_model_predict(xgb_model, window)
                future_predictions.append(prediction[0])
                last_window.append(prediction[0])
                last_window.pop(0)  # 移除窗口的第一个元素

            future_predictions = np.array(future_predictions)
            
            


        # 确保future_steps和测试集长度一致，取较短的长度
        comparison_length = min(self.future_steps, len(x_list_test))
        future_predictions = future_predictions[:comparison_length]
        x_list_test = x_list_test[:comparison_length]
        # 计算测试集上的MSE和MAE
        mse_test = np.mean((future_predictions - x_list_test) ** 2)
        mae_test = np.mean(np.abs(future_predictions - x_list_test))


        def save_svr_results(mae_train, mse_train, mae_test, mse_test):
            # 确保EXP-Details文件夹存在
            if not os.path.exists('EXP-Details'):
                os.makedirs('EXP-Details')

            # 定义文件路径
            file_path = f'EXP-Details/{self.csv_file_name}-SVR-mae+mse.txt'

            # 打开文件进行写入
            with open(file_path, 'w') as file:
                file.write("="*25 + f" SVR Results of {self.csv_file_name} " + "="*25 + "\n")
                file.write(f"MAE Train: {mae_train}\n")
                file.write(f"MSE Train: {mse_train}\n")
                file.write("-"*50 + "\n")
                file.write(f"MAE Future: {mae_test}\n")
                file.write(f"MSE Future: {mse_test}\n")
                file.write("="*50 + "\n")
                file.write("\n\n")

        save_svr_results(mae_train, mse_train, mae_test, mse_test)

        '''
        print("="*25 + f" SVR Results of {self.csv_file_name} " + "="*25)
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


                # 获取原始数据的最小和最大值
        min_value = np.min(value_list)
        max_value = np.max(value_list)
        # 反归一化训练集预测数据
        y_pred_train = y_pred_train * (max_value - min_value) + min_value
        # 反归一化未来预测数据
        future_predictions = future_predictions * (max_value - min_value) + min_value

        if(if_polt!=0):
            self.canva2_combined_update(value_list,y_pred_train,future_predictions,self.SVR_window_size)
        return y_pred_train , future_predictions
        

    
    def SVR_run_TrainPredcit_test(self, value_list, if_polt):
        
        self.SVR_window_size = int(self.SVR_window_size_rate * len(value_list))



        #--------------------检查窗口大小与数据量的关系-------------------# 
        if self.SVR_window_size_rate >= 0.6 :
            print(f'参数错误：模型窗口({self.SVR_window_size})大于训练集')
            return
        if self.SVR_window_size_rate >= 0.2:
            print(f'参数错误：模型窗口({self.SVR_window_size})大于验证集')
            return
        #--------------------检查窗口大小与数据量的关系-------------------# 
        


        #--------------------数据归一化-------------------# 
        value_list = self.preprocess_mimmax(value_list)
        #--------------------数据归一化-------------------# 



        #--------------------模型训练-------------------#
        X_train, y_train, X_val, y_val = self.SVR_prepare_data(value_list)
        xgb_model = self.SVR_model_train(X_train, y_train, X_val, y_val)
        #--------------------模型训练-------------------#



        print("Predicting... (｀・ω・´)")





        #--------------------预测训练集上数据-------------------#
        y_pred_train = []
        for window in X_train:
            prediction = self.SVR_model_predict(xgb_model, window.reshape(1, -1))
            y_pred_train.append(prediction[0])
        y_pred_train = np.array(y_pred_train)
        #--------------------预测训练集上数据-------------------#



        #--------------------计算训练集上的MSE和MAE-------------------#
        mse_train = np.mean((y_pred_train - y_train) ** 2)
        mae_train = np.mean(np.abs(y_pred_train - y_train))
        #--------------------计算训练集上的MSE和MAE-------------------#


        #--------------------分割训练集和测试集 8/10  2/10-------------------#
        x_list_train, x_list_test = train_test_split(value_list, test_size=0.20, shuffle=False)
        #--------------------分割训练集和测试集 8/10  2/10-------------------#



        # 计算80%位置的索引
        split_index = math.ceil(0.8 * len(x_list_train))
        # 从80%位置向前获取滑动窗口的数据
        # 预测未来 n 步
        self.future_steps = int(self.SVR_predict_rate * len(x_list_train))

        if self.future_steps > 0:
            last_window = x_list_train[split_index - self.SVR_window_size:split_index].tolist()
            future_predictions = []
            steps_to_predict = self.future_steps
            for _ in range(steps_to_predict):
                window = np.array(last_window[-self.SVR_window_size:]).reshape(1, -1)
                prediction = self.SVR_model_predict(xgb_model, window)
                future_predictions.append(prediction[0])
                last_window.append(prediction[0])
                last_window.pop(0)  # 移除窗口的第一个元素

            future_predictions = np.array(future_predictions)
            
            


        # 确保future_steps和测试集长度一致，取较短的长度
        comparison_length = min(self.future_steps, len(x_list_test))
        future_predictions = future_predictions[:comparison_length]
        x_list_test = x_list_test[:comparison_length]
        # 计算测试集上的MSE和MAE
        mse_test = np.mean((future_predictions - x_list_test) ** 2)
        mae_test = np.mean(np.abs(future_predictions - x_list_test))


        def save_svr_results(mae_train, mse_train, mae_test, mse_test):
            # 确保EXP-Details文件夹存在
            if not os.path.exists('EXP-Details'):
                os.makedirs('EXP-Details')

            # 定义文件路径
            file_path = f'EXP-Details/{self.csv_file_name}-SVR-mae+mse.txt'

            # 打开文件进行写入
            with open(file_path, 'w') as file:
                file.write("="*25 + f" SVR Results of {self.csv_file_name} " + "="*25 + "\n")
                file.write(f"MAE Train: {mae_train}\n")
                file.write(f"MSE Train: {mse_train}\n")
                file.write("-"*50 + "\n")
                file.write(f"MAE Future: {mae_test}\n")
                file.write(f"MSE Future: {mse_test}\n")
                file.write("="*50 + "\n")
                file.write("\n\n")

        save_svr_results(mae_train, mse_train, mae_test, mse_test)

        '''
        print("="*25 + f" SVR Results of {self.csv_file_name} " + "="*25)
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


                # 获取原始数据的最小和最大值
        min_value = np.min(value_list)
        max_value = np.max(value_list)
        # 反归一化训练集预测数据
        y_pred_train = y_pred_train * (max_value - min_value) + min_value
        # 反归一化未来预测数据
        future_predictions = future_predictions * (max_value - min_value) + min_value

        return y_pred_train , future_predictions
        


    def canva2_combined_update(self, y_actual, y_pred_on_actual, y_future_pred, window_size, split_index=0.8, title='SVR Plot of '):

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

        #plt.legend(fontsize=18, loc='upper left')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08)

        # 保存图像到指定路径
        save_path = f'Images/{self.csv_file_name}-SVR'
        plt.savefig(save_path)  # 保存图像
        #plt.show()


if __name__ == "__main__":
    SVR_itme = SVR_fun()
    value_list = SVR_itme.load_data(SVR_itme.csv_file_path)
    SVR_itme.SVR_run_TrainPredcit_test(value_list, 1)
