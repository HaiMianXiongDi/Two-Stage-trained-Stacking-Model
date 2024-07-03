from XGBoostFun import XGBoost_fun
from SVRFun import SVR_fun
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import math
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Stacking_fun():

    def __init__(self):
        self.XGB_item = XGBoost_fun()
        self.SVR_item = SVR_fun()
        self.window_size_rate = 0.1
        # Parameter configuration
        'Temp'
        'Exchange'
        'Electricity'
        self.csv_file_name = 'Temp'
        self.csv_file_path = 'Datasets/' + self.csv_file_name + '.csv'  # CSV file path
        self.file_column_index = 1 # Column index of the value in the file: 0,1,2,3,...

        self.XGB_item.csv_file_name = self.csv_file_name
        self.XGB_item.XGBoost_window_size_rate = self.window_size_rate
        
        self.SVR_item.csv_file_name = self.csv_file_name
        self.SVR_item.SVR_window_size_rate = self.window_size_rate

    def load_data(self, csv_file_path):
        try:
            data = pd.read_csv(csv_file_path)
            if data is not None and self.file_column_index < len(data.columns):
                value_list = data.iloc[:, self.file_column_index].tolist()
                # Check each element and try to convert if it's not a float
                value_list = [float(item) if not isinstance(item, float) else item for item in value_list]
                return value_list
            else:
                print("Data column error or file is empty")
        except pd.errors.ParserError as e:
            print(f"Error reading CSV file: {e}")
        return None

    def preprocess_mimmax(self, value_list):
        # Normalization
        value_min = np.min(value_list)
        value_max = np.max(value_list)
        value_list = ((value_list - value_min) / (value_max - value_min)).astype(np.float32)

        return value_list

    def canva2_combined_update(self, y_actual, y_pred_on_actual, y_future_pred, window_size, split_index=0.8, title='Stacking Plot of '):

        title = title + self.csv_file_name
        # Plot settings
        plt.figure(figsize=(14, 8))
        plt.title(title, fontsize=20)
        plt.xlabel("Timestamp", fontsize=20)
        plt.ylabel("Value", fontsize=20)
        plt.tick_params(labelsize=10)
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.5)

        # Plot the data retrieved from the database
        # plt.plot(x, y, label='Database Data', color='Green', linewidth=1.5)

        # Below is the original code from canva2_prdict_update, slightly modified
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

        # Save image to the specified path
        save_path = f'Images/{self.csv_file_name}-Stacking'
        plt.savefig(save_path)  # Save image
        # plt.show()

    def plot_with_bisector(self, y_actual, y_pred_on_actual, y_future_pred, window_size, split_index=0.8, title='Two-Stage Stacking Bisector Plot of '):
        
        title = title + self.csv_file_name

        # Combine prediction data and actual data for plotting
        y_pred_combined = np.concatenate((y_pred_on_actual, y_future_pred))
        y_actual_combined = y_actual[window_size:window_size + len(y_pred_combined)]

        fig, ax = plt.subplots(figsize=(6, 12))  
        scatter = ax.scatter(y_actual_combined, y_pred_combined, 
                    c=y_actual_combined, cmap='viridis', 
                    alpha=0.7, s=30, 
                    edgecolors='white', linewidths=0.5,
                    label='Predictions')
        
        # Add a thin colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(scatter, cax=cax)
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title, fontsize=15)
        ax.tick_params(labelsize=10)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        # Draw the ideal line
        min_value = min(min(y_actual_combined), min(y_pred_combined))
        max_value = max(max(y_actual_combined), max(y_pred_combined))
        x = np.linspace(min_value, max_value, 100)
        ax.plot(x, x, color='red', label='Ideal Line')

        plt.tight_layout()  # Automatically adjust subplot parameters to fill the entire figure area
        ax.legend()
        
        # Save image to the specified path
        save_path = f'Images/{self.csv_file_name}-Two-Stage Stacking-Bisector.png'
        plt.savefig(save_path, bbox_inches='tight')  # Save image, bbox_inches='tight' removes extra whitespace
        plt.show()  # Display image

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

        # Get the first 60% of the data as train_data
        train_data = value_list[:index_60_percent]
        # Get data from 60% to 80% as validation_data
        validation_data = value_list[index_60_percent:index_80_percent]

        # Convert time series data to supervised learning data
        X_train, y_train = self.create_lagged_features(train_data, self.window_size)
        X_val, y_val = self.create_lagged_features(validation_data, self.window_size)

        return X_train, y_train, X_val, y_val

    def Stacking_run_Predciting(self, value_list):
        
        value_list = self.preprocess_mimmax(value_list)
        self.window_size = int(self.window_size_rate * len(value_list))
  
        # Calculate the 80% position
        value_80 = len(value_list) * 0.8
        value_60 = len(value_list) * 0.6

        split_index_80 = math.ceil(len(value_list) * 0.8)
        split_index_60 = math.ceil(len(value_list) * 0.6)

        is_decimal_80 = value_80 != math.floor(value_80)
        is_decimal_60 = value_60 != math.floor(value_60)

        # Get the first 80% of the data
        value_list_80 = value_list[:split_index_80]
        # Get the first 60% of the data
        value_list_60 = value_list[:split_index_60]
        # Get the last 20% of the data (i.e., 80% to 100% of the data)
        value_list_20 = value_list[split_index_80:]
        # Get data from the beginning of self.window_size
        if(is_decimal_60):
            value_list_60_from_window = value_list_60[self.window_size+1:]
        else:
            value_list_60_from_window = value_list_60[self.window_size:]

        # To get the last 20% of the data in value_list_80, we need to calculate the corresponding index in value_list_80
        split_index_80_in_80 = math.ceil(len(value_list_80) * 0.8)
        # Get the last 20% of the data in value_list_80 (i.e., the part from 64% to 80% of the original list)
        value_list_80_last_20 = value_list_80[split_index_80_in_80:]

        xgboost_y_pred_train , xgboost_future_predictions = self.XGB_item.XGBoost_run_TrainPredcit(value_list_80, 0)
        svr_y_pred_train , svr_future_predictions = self.SVR_item.SVR_run_TrainPredcit(value_list_80, 0)
        # Stack the training set predictions of XGBOOST and SVR as new features
        '''
        stacked_features_future = np.column_stack((xgboost_future_predictions, svr_future_predictions))
        
        meta_model = LinearRegression()
        meta_model.fit(stacked_features_future, value_list_80_last_20)
        '''
        # Find the length of the shortest array
        min_length = min(len(xgboost_future_predictions), len(svr_future_predictions), len(value_list_80_last_20))

        # Align the lengths of xgboost and svr prediction results and value_list_80_last_20 by truncating
        xgboost_future_predictions_aligned = xgboost_future_predictions[:min_length]
        svr_future_predictions_aligned = svr_future_predictions[:min_length]
        value_list_80_last_20_aligned = value_list_80_last_20[:min_length]

        # Now we can safely stack and train
        stacked_features_future = np.column_stack((xgboost_future_predictions_aligned, svr_future_predictions_aligned))
        meta_model = LinearRegression()
        meta_model.fit(stacked_features_future, value_list_80_last_20_aligned)

        '''
        print("Length (value_list_80):", len(value_list_80))
        print("Length (value_list_20):", len(value_list_20))
        print("Length (split_index_80_in_80):", split_index_80_in_80)
        print("Length (value_list_80_last_20):", len(value_list_80_last_20))

        print("Length (xgboost_y_pred_train):", len(xgboost_y_pred_train))
        print("Length (xgboost_future_predictions):", len(xgboost_future_predictions))
        print("Length (svr_y_pred_train):", len(svr_y_pred_train))
        print("Length (svr_future_predictions):", len(svr_future_predictions))
        '''

        # ------------------ Predict future data ---------------------------#
        xgboost_y_pred_train , xgboost_future_predictions = self.XGB_item.XGBoost_run_TrainPredcit(value_list, 1)
        svr_y_pred_train , svr_future_predictions = self.SVR_item.SVR_run_TrainPredcit(value_list, 1)

        stacked_features_train = np.column_stack((xgboost_y_pred_train, svr_y_pred_train))
        stacked_features_future = np.column_stack((xgboost_future_predictions, svr_future_predictions))

        stacked_train_predictions = meta_model.predict(stacked_features_train)
        stacked_future_predictions = meta_model.predict(stacked_features_future)

        # self.canva2_combined_update(value_list,stacked_train_predictions,stacked_future_predictions,self.XGB_item.XGBoost_window_size)

        mse_train = np.mean((stacked_train_predictions - value_list_60_from_window ) ** 2)
        mae_train = np.mean(np.abs(stacked_train_predictions - value_list_60_from_window ))	
        mse_test = np.mean((stacked_future_predictions - value_list_20) ** 2)
        mae_test = np.mean(np.abs(stacked_future_predictions - value_list_20))

        def save_stk_results(mae_train, mse_train, mae_test, mse_test):
            # Ensure the EXP-Details folder exists
            if not os.path.exists('EXP-Details'):
                os.makedirs('EXP-Details')

            # Define file path
            file_path = f'EXP-Details/{self.csv_file_name}-Stacking-mae+mse.txt'

            # Open file for writing
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
        self.canva2_combined_update(value_list, stacked_train_predictions, stacked_future_predictions, self.XGB_item.XGBoost_window_size)
        self.plot_with_bisector(value_list, stacked_train_predictions, stacked_future_predictions, self.XGB_item.XGBoost_window_size)
        print("Experiment all finished! Check the results in /EXP-Details and /Images")
        return

if __name__ == "__main__":
    STK_item = Stacking_fun()
    value_list = STK_item.load_data(STK_item.csv_file_path)
    STK_item.Stacking_run_Predciting(value_list)
