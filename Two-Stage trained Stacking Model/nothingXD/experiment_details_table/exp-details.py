'''
\documentclass{standalone}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}

\begin{document}

\begin{tabular}{c|cc|cc|cc} 
\textbf{Methods} & \multicolumn{2}{c|}{\textbf{XGBoost}} & \multicolumn{2}{c|}{\textbf{SVR}} & \multicolumn{2}{c}{\textbf{Stacking}} \\
\textbf{Sets} & \textbf{Training} & \textbf{Test} & \textbf{Training} & \textbf{Test} & \textbf{Training} & \textbf{Test} \\
\textbf{Metric} & \textbf{MAE MSE}  & \textbf{MAE MSE} & \textbf{MAE MSE} & \textbf{MAE MSE} & \textbf{MAE MSE} & \textbf{MAE MSE} \\
\hline 
Electricity & \textbf{0.0179}  \textbf{0.0005} &  0.1116  0.0238 & 0.0425  0.0025 &  0.1118  0.0186 & 0.0438  0.0029 &  \textbf{0.1057}  \textbf{0.0173} \\

Temp & \textbf{0.0107}  \textbf{0.0001} &  0.1118  0.0190 & 0.0630  0.0051 &  0.0991  0.0148 & 0.0614  0.0058 &  \textbf{0.0848}  \textbf{0.0113} \\

Exchange & \textbf{0.0041}  \textbf{3.1e-05} &  0.1232  0.0270 & 0.0399  0.0023 &  0.2209  0.0889 & 0.1805  0.0588 &  \textbf{0.0834}  \textbf{0.0122} \\
\end{tabular}

\end{document}
'''