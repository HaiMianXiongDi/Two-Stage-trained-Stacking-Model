# Two-Stage-trained-Stacking-Model
The experiment project of Two-Stage trained Stacking Model

## environment setting

conda create -n env-StageTwoStacking python=3.9  
conda activate env-StageTwoStacking  

conda install numpy  
conda install scikit-learn  
conda install matplotlib  
conda install pandas    
conda install xgboost  

## how to use
To predict your data, please place your CSV file into the 'Datasets' folder. Then, open StackingFun-TwoStage and modify the 'self.csv_file_name' in the 'init' function to match the name of your file. The project comes with three datasets by default:

'Temp'  
'Exchange'  
'Electricity'  

These are the datasets used in the paper.
Another parameter, self.file_column_index, represents which column in the CSV file contains the value of the time series data. By default, it is set to 1, meaning that the second column of the CSV is selected as the value column for the time series data.

After making these adjustments, simply run the entire 'StackingFun-TwoStage.py' script. Once the console outputs the message 'Experiment all finished!', it indicates that the training and prediction processes have been completed.

You can check the detailed experimental results in the 'EXP-Details' folder and view the visualization of the experiments in the 'Images' folder.

