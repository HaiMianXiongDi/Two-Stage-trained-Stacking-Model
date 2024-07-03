from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SVR_fun():
#---------------------------------------------------------------------#

    def __init__(self):

        self.SVR_train_size = 0.8  # Size of training set + validation set
        self.num_boost_round = 100
        self.SVR_window_size_rate = 0.1  # Length of original data * SVR_window_size_rate
        self.SVR_predict_rate = 0.2

        # Parameter configuration
        self.kernel = 'rbf'
        self.C = 1.5
        self.epsilon = 0.1
        # Parameter configuration
        'Temp'
        'Exchange'
        'Electricity'
        self.csv_file_name = 'Exchange'
        self.csv_file_path = 'Datasets/' + self.csv_file_name + '.csv'  # CSV file path
        self.file_column_index = 1  # Value column in file: 0, 1, 2, 3, ...

    def SVR_create_lagged_features(self, data, SVR_window_size):
        X, y = [], []
        for i in range(SVR_window_size, len(data)):
            X.append(data[i - SVR_window_size:i])
            y.append(data[i])
        X, y = np.array(X), np.array(y)
        return X, y

    def SVR_prepare_data(self, Source_data):
        value_list = Source_data

        index_60_percent = int(len(value_list) * 0.6)
        index_80_percent = int(len(value_list) * 0.8)

        # Get the first 60% of data as train_data
        train_data = value_list[:index_60_percent]
        # From 60% to 80% as validation_data
        validation_data = value_list[index_60_percent:index_80_percent]

        # Convert time series data to supervised learning data
        X_train, y_train = self.SVR_create_lagged_features(train_data, self.SVR_window_size)
        X_val, y_val = self.SVR_create_lagged_features(validation_data, self.SVR_window_size)

        return X_train, y_train, X_val, y_val

    def SVR_model_train(self, x_train, y_train, x_val, y_val):
        # Create SVR model instance
        model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, verbose=True)
        model.fit(x_train, y_train)
        # Return the trained model
        return model

    def SVR_model_predict(self, svm_model, X_data):
        predictions = svm_model.predict(X_data)
        return predictions

    def load_data(self, csv_file_path):
        try:
            data = pd.read_csv(csv_file_path)
            if data is not None and self.file_column_index < len(data.columns):
                value_list = data.iloc[:, self.file_column_index].tolist()
                # Check each element, if it is not a float, try to convert it
                value_list = [float(item) if not isinstance(item, float) else item for item in value_list]
                return value_list
            else:
                print("Error: Incorrect data column or empty file")
        except pd.errors.ParserError as e:
            print(f"Error reading CSV file: {e}")
        return None

    def preprocess_mimmax(self, value_list):
        # Normalization
        value_min = np.min(value_list)
        value_max = np.max(value_list)
        value_list = ((value_list - value_min) / (value_max - value_min)).astype(np.float32)
        return value_list

    def SVR_run_TrainPredcit(self, value_list, if_polt):

        self.SVR_window_size = int(self.SVR_window_size_rate * len(value_list))

        #-------------------- Check the relationship between window size and data volume -------------------# 
        if self.SVR_window_size_rate >= 0.6:
            print(f'Parameter Error: Model window ({self.SVR_window_size}) is greater than the training set')
            return
        if self.SVR_window_size_rate >= 0.2:
            print(f'Parameter Error: Model window ({self.SVR_window_size}) is greater than the validation set')
            return
        #-------------------- Check the relationship between window size and data volume -------------------# 

        #-------------------- Data Normalization -------------------# 
        value_list = self.preprocess_mimmax(value_list)
        #-------------------- Data Normalization -------------------# 

        #-------------------- Model Training -------------------#
        X_train, y_train, X_val, y_val = self.SVR_prepare_data(value_list)
        xgb_model = self.SVR_model_train(X_train, y_train, X_val, y_val)
        #-------------------- Model Training -------------------#

        print("Predicting... (｀・ω・´)")

        #-------------------- Predict data on the training set -------------------#
        y_pred_train = []
        for window in X_train:
            prediction = self.SVR_model_predict(xgb_model, window.reshape(1, -1))
            y_pred_train.append(prediction[0])
        y_pred_train = np.array(y_pred_train)
        #-------------------- Predict data on the training set -------------------#

        #-------------------- Calculate MSE and MAE on the training set -------------------#
        mse_train = np.mean((y_pred_train - y_train) ** 2)
        mae_train = np.mean(np.abs(y_pred_train - y_train))
        #-------------------- Calculate MSE and MAE on the training set -------------------#

        #-------------------- Split the training set and test set 8/10 2/10 -------------------#
        x_list_train, x_list_test = train_test_split(value_list, test_size=0.20, shuffle=False)
        #-------------------- Split the training set and test set 8/10 2/10 -------------------#

        # Predict future n steps
        self.future_steps = int(self.SVR_predict_rate * len(value_list))
        if self.future_steps > 0:
            last_window = x_list_train[-self.SVR_window_size:].tolist()  # Use the last window of training data
            future_predictions = []
            steps_to_predict = self.future_steps
            for _ in range(steps_to_predict):
                window = np.array(last_window[-self.SVR_window_size:]).reshape(1, -1)
                prediction = self.SVR_model_predict(xgb_model, window)
                future_predictions.append(prediction[0])
                last_window.append(prediction[0])
                last_window.pop(0)  # Remove the first element of the window

            future_predictions = np.array(future_predictions)

        # Ensure future_steps and test set length are consistent, take the shorter length
        comparison_length = min(self.future_steps, len(x_list_test))
        future_predictions = future_predictions[:comparison_length]
        x_list_test = x_list_test[:comparison_length]
        # Calculate MSE and MAE on the test set
        mse_test = np.mean((future_predictions - x_list_test) ** 2)
        mae_test = np.mean(np.abs(future_predictions - x_list_test))

        def save_svr_results(mae_train, mse_train, mae_test, mse_test):
            # Ensure the EXP-Details folder exists
            if not os.path.exists('EXP-Details'):
                os.makedirs('EXP-Details')

            # Define the file path
            file_path = f'EXP-Details/{self.csv_file_name}-SVR-mae+mse.txt'

            # Open the file for writing
            with open(file_path, 'w') as file:
                file.write("=" * 25 + f" SVR Results of {self.csv_file_name} " + "=" * 25 + "\n")
                file.write(f"MAE Train: {mae_train}\n")
                file.write(f"MSE Train: {mse_train}\n")
                file.write("-" * 50 + "\n")
                file.write(f"MAE Future: {mae_test}\n")
                file.write(f"MSE Future: {mse_test}\n")
                file.write("=" * 50 + "\n")
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

        # Get the minimum and maximum values of the original data
        min_value = np.min(value_list)
        max_value = np.max(value_list)
        # Denormalize training set prediction data
        y_pred_train = y_pred_train * (max_value - min_value) + min_value
        # Denormalize future prediction data
        future_predictions = future_predictions * (max_value - min_value) + min_value

        if(if_polt != 0):
            self.canva2_combined_update(value_list, y_pred_train, future_predictions, self.SVR_window_size)
            self.plot_with_bisector(value_list, y_pred_train, future_predictions, self.SVR_window_size)
        return y_pred_train, future_predictions

    def SVR_run_TrainPredcit_test(self, value_list, if_polt):

        self.SVR_window_size = int(self.SVR_window_size_rate * len(value_list))

        #-------------------- Check the relationship between window size and data volume -------------------# 
        if self.SVR_window_size_rate >= 0.6:
            print(f'Parameter Error: Model window ({self.SVR_window_size}) is greater than the training set')
            return
        if self.SVR_window_size_rate >= 0.2:
            print(f'Parameter Error: Model window ({self.SVR_window_size}) is greater than the validation set')
            return
        #-------------------- Check the relationship between window size and data volume -------------------# 

        #-------------------- Data Normalization -------------------# 
        value_list = self.preprocess_mimmax(value_list)
        #-------------------- Data Normalization -------------------# 

        #-------------------- Model Training -------------------#
        X_train, y_train, X_val, y_val = self.SVR_prepare_data(value_list)
        xgb_model = self.SVR_model_train(X_train, y_train, X_val, y_val)
        #-------------------- Model Training -------------------#

        print("Predicting... (｀・ω・´)")

        #-------------------- Predict data on the training set -------------------#
        y_pred_train = []
        for window in X_train:
            prediction = self.SVR_model_predict(xgb_model, window.reshape(1, -1))
            y_pred_train.append(prediction[0])
        y_pred_train = np.array(y_pred_train)
        #-------------------- Predict data on the training set -------------------#

        #-------------------- Calculate MSE and MAE on the training set -------------------#
        mse_train = np.mean((y_pred_train - y_train) ** 2)
        mae_train = np.mean(np.abs(y_pred_train - y_train))
        #-------------------- Calculate MSE and MAE on the training set -------------------#

        #-------------------- Split the training set and test set 8/10 2/10 -------------------#
        x_list_train, x_list_test = train_test_split(value_list, test_size=0.20, shuffle=False)
        #-------------------- Split the training set and test set 8/10 2/10 -------------------#

        # Calculate the index at 80%
        split_index = math.ceil(0.8 * len(x_list_train))
        # Get sliding window data from 80% position
        # Predict future n steps
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
                last_window.pop(0)  # Remove the first element of the window

            future_predictions = np.array(future_predictions)

        # Ensure future_steps and test set length are consistent, take the shorter length
        comparison_length = min(self.future_steps, len(x_list_test))
        future_predictions = future_predictions[:comparison_length]
        x_list_test = x_list_test[:comparison_length]
        # Calculate MSE and MAE on the test set
        mse_test = np.mean((future_predictions - x_list_test) ** 2)
        mae_test = np.mean(np.abs(future_predictions - x_list_test))

        def save_svr_results(mae_train, mse_train, mae_test, mse_test):
            # Ensure the EXP-Details folder exists
            if not os.path.exists('EXP-Details'):
                os.makedirs('EXP-Details')

            # Define the file path
            file_path = f'EXP-Details/{self.csv_file_name}-SVR-mae+mse.txt'

            # Open the file for writing
            with open(file_path, 'w') as file:
                file.write("=" * 25 + f" SVR Results of {self.csv_file_name} " + "=" * 25 + "\n")
                file.write(f"MAE Train: {mae_train}\n")
                file.write(f"MSE Train: {mse_train}\n")
                file.write("-" * 50 + "\n")
                file.write(f"MAE Future: {mae_test}\n")
                file.write(f"MSE Future: {mse_test}\n")
                file.write("=" * 50 + "\n")
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

        # Get the minimum and maximum values of the original data
        min_value = np.min(value_list)
        max_value = np.max(value_list)
        # Denormalize training set prediction data
        y_pred_train = y_pred_train * (max_value - min_value) + min_value
        # Denormalize future prediction data
        future_predictions = future_predictions * (max_value - min_value) + min_value

        return y_pred_train, future_predictions

    def canva2_combined_update(self, y_actual, y_pred_on_actual, y_future_pred, window_size, split_index=0.8, title='SVR Plot of '):

        title = title + self.csv_file_name
        # Plot settings
        plt.figure(figsize=(14, 8))
        plt.title(title, fontsize=20)
        plt.xlabel("Timestamp", fontsize=20)
        plt.ylabel("Value", fontsize=20)
        plt.tick_params(labelsize=10)
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.5)

        # Plot the data obtained from the database
        # plt.plot(x, y, label='Database Data', color='Green', linewidth=1.5)

        # The following code is slightly modified from the original canva2_prdict_update
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

        # Save the image to the specified path
        save_path = f'Images/{self.csv_file_name}-SVR'
        plt.savefig(save_path)  # Save the image
        # plt.show()

    def plot_with_bisector(self, y_actual, y_pred_on_actual, y_future_pred, window_size, split_index=0.8, title='SVR Bisector Plot of '):

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

        # Add a slender colorbar
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

        plt.tight_layout()  # Automatically adjust subplot parameters to fill the entire image area
        ax.legend()

        # Save the image to the specified path
        save_path = f'Images/{self.csv_file_name}-SVR-Bisector.png'
        plt.savefig(save_path, bbox_inches='tight')  # Save the image, bbox_inches='tight' removes extra white space
        plt.show()  # Show the image


if __name__ == "__main__":
    SVR_itme = SVR_fun()
    value_list = SVR_itme.load_data(SVR_itme.csv_file_path)
    SVR_itme.SVR_run_TrainPredcit(value_list, 1)
