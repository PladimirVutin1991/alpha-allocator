\documentclass[11pt]{article}

% --- PACKAGES ---
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{a4paper, margin=0.85in} % FIXED: Optimized margins (not too big)
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{float}
\usepackage{titlesec}
\usepackage{amsmath}
\usepackage{xcolor}
% --- CUSTOMIZE ABSTRACT SIZE ---
\usepackage{abstract}
\renewcommand{\abstractnamefont}{\normalfont\Large\bfseries} % Makes "Abstract" title larger
\renewcommand{\abstracttextfont}{\normalfont\large}          % Makes the text itself larger
\usepackage{caption}

% --- HYPERLINK SETUP ---
\hypersetup{
    colorlinks=true,
    linkcolor=black, % Black looks more professional for TOC
    filecolor=blue,      
    urlcolor=blue,
    citecolor=blue
}

% --- TITLE INFO ---
\title{\textbf{Alpha Allocator: Portfolio Optimization via Sector-Relative Machine Learning}}
\author{Vlad Sandrovschi \\ \texttt{vlad.sandrovschi@unil.ch}}
\date{January 11, 2026}

\begin{document}

\maketitle



\
\\
\
\\
\

\begin{abstract}
\large Constructing an equity portfolio that consistently outperforms the S\&P 500 benchmark is a persistent challenge in quantitative finance, often hampered by low signal-to-noise ratios and broad market regime shifts. Traditional absolute return predictions frequently fail by merely capturing market beta rather than true alpha. This project addresses this limitation by developing a machine learning pipeline designed to identify high-conviction ``Buy'' signals through a novel sector-relative performance ranking approach.

Leveraging a dataset of approximately 320 US equities across 11 GICS sectors spanning 2015--2025, I engineered a set of ``Smart Beta'' features focused on momentum, relative strength, and trend distance. A \textbf{Histogram Gradient Boosting Classifier} was trained to predict whether a stock would outperform its specific sector median over a 3-month horizon, effectively neutralizing broad market movements. By dynamically optimizing the decision threshold to target the top 10\% of confidence scores, the model achieved a \textbf{54.6\% Win Rate} on the out-of-sample test set (2024--2025), significantly exceeding the random baseline.

These predictive signals drive a dynamic portfolio simulation engine implementing a ``Zero-Tolerance'' strategy to high-conviction assets while completely divesting from low-conviction holdings. The resulting Monte Carlo simulations demonstrate that this machine learning overlay generates significant alpha, producing a portfolio that consistently diverges positively from the passive benchmark.
\end{abstract}

\newpage
\tableofcontents
\newpage

\section{Introduction}

\subsection{Motivation}

As a member of a student investment association at HEC Lausanne, we actively manage a fund of our own money through equity trades. We pitch and identify likely stocks to outperform the S\&P 500. It is a challenging task to find such companies given the size of the market. Stocks within the same sector often diverge significantly based on idiosyncratic factors. Traditional screening methods (e.g., P/E ratio cutoffs) are static and fail to capture non-linear relationships in market data.

This project aims to bridge the gap between Data Science and Asset Management by building a Machine Learning (ML) classifier that tries to predict if a certain stock beats the benchmark over a 3-month period. It is not merely a stock predictor but a tool that integrates machine learning predictions for stock picking with an asset allocation dashboard (Black-Litterman tilting) to create an investment strategy.

\subsection{Problem Statement}
Predicting raw stock returns is notoriously difficult due to low signal-to-noise ratios. A model trained to predict if a stock beats the S&P 500 often fails during market downturns (where no stock beats the benchmark) or tech bubbles (where only Tech beats the benchmark). This creates a bias where the model simply learns to be ``Long Tech'' or ``Short Energy.''

\subsection{Objectives}
\begin{itemize}
    \item \textbf{Data Engineering:} Construct a robust pipeline handling $\sim$320 stocks over 10 years, adjusting for sector-specific nuances.
    \item \textbf{Model Optimization:} Classification in a sector-relative ranking, targeting the top decile of performers.
    \item \textbf{Simulation:} Validate the strategy using a Monte Carlo simulation that compares the ML-enhanced portfolio against a standard passive benchmark.
    \item \textbf{Dashboard:} Visualize the results in an interactive dashboard for user use.
\end{itemize}

\section{Research Question \& Literature}

\textbf{Research Question:} \textit{Can a ML model, trained on sector-relative technical indicators, identify stocks that statistically outperform a passive market-cap-weighted benchmark?}

\subsection{Related Work}
\begin{itemize}
    \item \textbf{Fama \& French (1993):} Established that specific ``factors'' (like size and value) drive returns. I extend this concept by using non-linear ML models to find dynamic price-action factors rather than just static fundamental ratios.
    \item \textbf{Gu, Kelly, \& Xiu (2020):} Demonstrated that machine learning models, specifically decision trees and neural networks, outperform standard econometric regressions in asset pricing. This research validates my choice of tree-based models (Gradient Boosting) over simple linear regressions.
    \item \textbf{Relative Strength (Levy, 1967):} The concept that strong assets tend to outperform weak assets over time. I modernize this by calculating relative strength \textit{within} sectors (e.g., comparing Apple to Microsoft) rather than across the whole market to isolate alpha.
\end{itemize}

\section{Methodology}

\subsection{Data Acquisition}
I utilized the \texttt{yfinance} API to ingest daily adjusted closing prices for a diversified universe of 328 US equities and 8 Macro ETFs (Bonds, Commodities, Real Estate) from January 1, 2015, to December 30, 2025.

To ensure realistic portfolio construction, I segmented the chosen stocks into 11 sectors based on the \textbf{Global Industry Classification Standard (GICS)} (e.g., Tech, Financials, Healthcare). This segmentation is critical for the feature engineering phase.

\subsection{Feature Engineering}
A ``Smart Beta'' feature set was created to capture price dynamics relative to the market:
\begin{enumerate}
    \item \textbf{Momentum:} 1-month (\texttt{Ret\_1M}) and 3-month (\texttt{Ret\_3M}) rolling returns.
    \item \textbf{Relative Strength Index (RSI):} A 14-day technical indicator that measures the speed and change of price movements. It ranges from 0 to 100 and helps identify overbought or oversold conditions.
    \item \textbf{Trend Distance (\texttt{Trend\_SMA}):} The percentage distance of the current price from its 50-day Simple Moving Average. Positive values indicate an uptrend.
    \item \textbf{Market Relative Strength (\texttt{Rel\_Str\_3M}):} The stock's performance compared to the S\&P 500 performance over the last quarter.
\end{enumerate}

\subsection{Sector-Relative Target}
Initial experiments using a raw target (\textit{``Did stock beat S\&P 500?''}) yielded poor precision ($\sim$40\%) because the target was dominated by market beta (possibly the overall market direction into bullish tech stocks).

I then pivoted to a Sector-Relative Target. The target variable $y$ is defined as:
\[
y_{i,t} = 
\begin{cases} 
1 & \text{if } Return_{i,t+63} > \text{Median}(Return_{Sector, t+63}) \\
0 & \text{else}
\end{cases}
\]
This ``curves the grades,'' forcing the model to find the best Utility stock among Utilities or the best Tech stock among Tech. This guarantees a balanced dataset (50\% winners, 50\% losers) regardless of market conditions.

\subsection{Model Architecture}
I selected the \textbf{Histogram Gradient Boosting Classifier} (\texttt{HistGradientBoostingClassifier} from scikit-learn).
\begin{itemize}
    \item \textbf{Why Gradient Boosting?} Unlike Random Forests, which build trees independently, Gradient Boosting builds trees sequentially to correct the errors of previous trees. This makes it superior for detecting subtle signals in noisy financial data.
    \item \textbf{Training Regime:}
    \begin{itemize}
        \item \textbf{Training:} 2015--2023 (Learning historical patterns).
        \item \textbf{Testing:} 2024--2025 (Out-of-sample validation).
        \item \textbf{Optimization:} I used \texttt{RandomizedSearchCV} to find the best hyperparameters. This method randomly samples parameters from a grid, which is faster than checking every combination. I tuned: \texttt{learning\_rate}, \texttt{max\_depth}, and \texttt{min\_samples\_leaf}.
    \end{itemize}
\end{itemize}

\subsection{Strategic Allocation Profiles}
The dashboard allows users to select distinct risk profiles (Growth, Balanced, Income). These profiles define the base asset allocation before the machine learning model applies its active weights.

The portfolio is constructed using a robust set of underlying assets to ensure diversification:
\begin{itemize}
    \item \textbf{Equities (Active):} A universe of $\sim$320 stocks across all 11 GICS sectors.
    \item \textbf{Fixed Income (Passive):} A diversified bond mix using \textbf{AGG} (US Core Bonds), \textbf{LQD} (Investment Grade Corporate), and \textbf{HYG} (High Yield).
    \item \textbf{Alternatives (Passive):} Inflation hedges including \textbf{GLD} (Gold), \textbf{DBC} (Commodities Index), and \textbf{VNQ} (Real Estate REITs).
    \item \textbf{Liquidity:} \textbf{SHV} (Short-Term US Treasury) for risk-free stability.
\end{itemize}

\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Profile} & \textbf{Equities} & \textbf{Fixed Income} & \textbf{Liquidity} & \textbf{Alternatives} \\
\midrule
\textbf{Growth} & 70\% & 20\% & 5\% & 5\% \\
\textbf{Balanced} & 50\% & 40\% & 5\% & 5\% \\
\textbf{Income} & 30\% & 60\% & 5\% & 5\% \\
\bottomrule
\end{tabular}
\caption{Strategic asset allocation mixes.}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.32\textwidth]{pie_growth.png}
    \includegraphics[width=0.32\textwidth]{pie_balanced.png}
    \includegraphics[width=0.32\textwidth]{pie_income.png}
    \caption{Strategic asset allocation visual breakdown for Growth, Balanced, and Income profiles.}
\end{figure}

\subsection{Portfolio Construction \& Monte Carlo}
The model outputs a probability score (Confidence). Instead of buying all stocks with $>50\%$ probability, I implement a \textbf{Top Decile Strategy}:
\begin{enumerate}
    \item \textbf{Filter:} Only consider stocks in the top 10--20\% of confidence scores.
    \item \textbf{Aggressive Weighting:}
    \begin{itemize}
        \item \textbf{High Conviction (Signal 1):} Weight multiplier = \textbf{5.0x} (Overweight).
        \item \textbf{Low Conviction (Signal 0):} Weight multiplier = \textbf{0.0x} (Sell/Avoid).
    \end{itemize}
    \item \textbf{Monte Carlo Simulation:} To project future performance, the system executes a Monte Carlo simulation generating 200 independent price paths over a 252-day trading horizon. The simulation relies on \texttt{numpy.random.multivariate\_normal} to generate returns based on the historical covariance matrix of the selected assets. This methodology accounts for the correlation between assets (e.g., if Tech stocks crash, they likely move together), ensuring the risk projections are mathematically consistent with historical inter-asset relationships.
\end{enumerate}

\section{Implementation}

The project is structured as a modular Python pipeline in the \texttt{src/} directory.

\subsection{Key Technical Decisions}
\begin{itemize}
    \item \textbf{Dynamic Thresholding:} In \texttt{optimize\_model.py}, I implemented logic to dynamically scan the validation set for the probability threshold that maximizes precision.
    \item \textbf{Survivorship Bias Mitigation:} In \texttt{portfolio\_simulation.py}, the code dynamically cleans the asset universe. If a stock (e.g., UBER) did not exist in 2016, the covariance matrix calculation automatically excludes it or trims the simulation window to valid data points to prevent mathematical convergence errors.
    \item \textbf{Data Leakage Prevention:} Features are calculated using \textit{lagged} data (available at time $t$), while targets look forward to $t+63$. The train/test split is strictly temporal (no shuffling).
\end{itemize}

\subsection{Software Stack}
\textbf{Pandas/NumPy} for data manipulation, \textbf{Scikit-Learn} for Gradient Boosting, \textbf{Plotly} for interactive charts, and \textbf{Streamlit} for the web dashboard.

\subsection{Dashboard Interface}
The Streamlit dashboard serves as the control center. It allows users to visualize portfolio composition, run simulations, and audit model accuracy.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\textwidth]{dashboard.png}
    \caption{The Alpha Allocator Dashboard Interface.}
\end{figure}

\textbf{Key Feature: Conviction Threshold Slider} \\
The ``Conviction Threshold'' slider controls the risk tolerance of the AI model.
\begin{itemize}
    \item \textbf{Higher Threshold (e.g., 0.60):} The model forces extremely selective trading (high confidence only), increasing precision but reducing diversification.
    \item \textbf{Lower Threshold (e.g., 0.51):} The model trades more broadly, increasing diversification but potentially lowering the statistical edge per trade.
\end{itemize}

\section{Codebase \& Reproducibility}

\subsection{How to Run}
\begin{enumerate}
    \item \textbf{Environment:} The project includes a \texttt{requirements.txt} file listing dependencies.
    \item \textbf{Execution:} Run \texttt{python main.py}. This script automatically checks dependencies, runs data processing, trains the model, and launches the dashboard.
\end{enumerate}

\subsection{Reproducibility Features}
\begin{itemize}
    \item \textbf{Random State:} Fixed to \texttt{42} in all ML models and simulations.
    \item \textbf{Asset Handling:} Robust error handling for \texttt{yfinance} downloads prevents crashes if a ticker is delisted.
\end{itemize}

\section{Results}

\subsection{Model Performance}
The Gradient Boosting model, utilizing the Sector-Relative target, achieved impressive results on the 2024--2025 out-of-sample test set.
\begin{itemize}
    \item \textbf{Global Accuracy:} $\sim$52\%
    \item \textbf{Precision (Win Rate):} \textbf{54.6\%} on the Top 10\% high-confidence picks.
\end{itemize}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{model_audit.png}
    \caption{Model Audit showing Confusion Matrix and Outcome Distribution.}
\end{figure}

\subsection{Simulation Results (Alpha Generation)}
The Monte Carlo simulation (Figure 7) demonstrates the impact of this precision for a \textbf{Balanced Risk Profile}. By concentrating capital into the top decile of stocks and aggressively cutting losers, the \textbf{Active Portfolio (Blue)} decently exceeds the \textbf{Passive Benchmark (Red)}.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{simulation_balanced.png}
    \caption{Portfolio Simulation (Balanced Risk Profile)}
\end{figure}

\section{Conclusion}

\subsection{Summary}
The allocator successfully demonstrates that Machine Learning can extract alpha from public market data if the problem is framed correctly. The shift from ``Absolute Prediction'' to ``Sector-Relative Ranking'' was the turning point, allowing the model to identify quality assets regardless of broader market conditions.

\subsection{Limitations}
\begin{itemize}
    \item \textbf{Transaction Costs:} The simulation assumes free trading. High turnover could erode alpha.
    \item \textbf{Market Regime:} The test set (2024--2025) was largely a bull market. Performance in a severe recession remains to be fully stress-tested.
\end{itemize}

\subsection{Future Work}
\begin{itemize}
    \item \textbf{Sentiment Analysis:} Integrating NLP scores from financial news.
    \item \textbf{Risk Parity:} Implementing sophisticated weighting schemes beyond simple multipliers.
\end{itemize}

\newpage
\begin{thebibliography}{9}

\bibitem{fama1993}
Fama, E. F., \& French, K. R. (1993).
Common risk factors in the returns on stocks and bonds.
\textit{Journal of Financial Economics}, 33(1), 3--56.

\bibitem{gu2020}
Gu, S., Kelly, B., \& Xiu, D. (2020).
Empirical Asset Pricing via Machine Learning.
\textit{The Review of Financial Studies}, 33(5), 2223--2273.

\bibitem{pedregosa2011}
Pedregosa, F., et al. (2011).
Scikit-learn: Machine Learning in Python.
\textit{Journal of Machine Learning Research}, 12, 2825--2830.

\bibitem{levy1967}
Levy, R. A. (1967).
Relative Strength as a Criterion for Investment Selection.
\textit{The Journal of Finance}, 22(4), 595--610.

\end{thebibliography}

\section*{Appendix: AI Tools Usage}
This project utilized AI assistance (Gemini) for:
1. \textbf{Debugging:} Resolving \texttt{numpy.linalg.LinAlgError} (SVD convergence) in the simulation.
2. \textbf{Refactoring:} Optimizing the \texttt{data\_processing.py} loop.
3. \textbf{Visualization:} Generating Plotly code for the simulation chart.
4. \textbf{Research Data}: Searching relevant research that could be employed into the ML model.

\end{document}