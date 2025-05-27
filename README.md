# RNN_Stock_Price_Prediction
## Objective

The objective of this assignment is to try and predict the stock prices using historical data from four companies IBM (IBM), Google (GOOGL), Amazon (AMZN), and Microsoft (MSFT).
We use four different companies because they belong to the same sector: Technology. Using data from all four companies may improve the performance of the model. This way, we can capture the broader market sentiment.
The problem statement for this assignment can be summarised as follows:
Given the stock prices of Amazon, Google, IBM, and Microsoft for a set number of days, predict the stock price of these companies after that window.


## Business Value


Data related to stock markets lends itself well to modeling using RNNs due to its sequential nature. We can keep track of opening prices, closing prices, highest prices, and so on for a long period of time as these values are generated every working day. The patterns observed in this data can then be used to predict the future direction in which stock prices are expected to move. Analyzing this data can be interesting in itself, but it also has a financial incentive as accurate predictions can lead to massive profits.


## Data Description

You have been provided with four CSV files corresponding to four stocks: AMZN, GOOGL, IBM, and MSFT. The files contain historical data that were gathered from the websites of the stock markets where these companies are listed: NYSE and NASDAQ. The columns in all four files are identical. Let's take a look at them:

<b>Date</b>: The values in this column specify the date on which the values were recorded. In all four files, the dates range from Jaunary 1, 2006 to January 1, 2018.
<b>Open:</b> The values in this column specify the stock price on a given date when the stock market opens.
<b>High:</b> The values in this column specify the highest stock price achieved by a stock on a given date.
<b>Low:</b> The values in this column specify the lowest stock price achieved by a stock on a given date.
<b>Close:</b> The values in this column specify the stock price on a given date when the stock market closes.
<b>Volume:</b> The values in this column specify the total number of shares traded on a given date.
<b>Name:</b> This column gives the official name of the stock as used in the stock market.

There are 3019 records in each data set. The file names are of the format \<company_name>_stock_data.csv.


<br>
<br>

## Conclusion

In this assignment, we built and evaluated various RNN models to predict stock prices for AMAZON, GOOGLE, IBM, MICROSOFT.

Please find below the step by step approch we took to Build an efficient and stable model
1. Data Preparation:
   - We successfully gathered stock data for four major tech companies: Amazon, Google, IBM, and Microsoft.

   - The dataset comprised 3018 rows and 21 columns, with complete time-series data, after handling missing values and converting Date columns to datetime objects. A MinMaxScaler with windowed DataFrames using partial_fit is applied to avoid data leakage and normalize stock price features for better convergence of the neural network.

  - Our exploratory data analysis revealed a noticeable correlation among all companies closing prices, indicating that these companies tend to follow similar market trends (maybe due to their presence in the same industry sector).

  - Additionally, we found that the volume distributions were right-skewed, suggesting that there are occasional spikes in trading activity, often corresponding to significant market events.

2. Window Size Selection:
   - After analyzing the time series patterns, we selected a window size of 23 days (approximately one Business Month of trading days) which captures meaningful business cycles in the stock market.
   - This window size provided a good balance between capturing long-term trends without making the model too complex.

3. Simple RNN vs Advanced RNN Models:
   - The advanced RNN models (LSTM/GRU) were consistently performed well on simple RNN models in terms of MSE, MAE, and R² metrics.
   - This performance difference highlights the advantage of LSTM/GRU in capturing long-term dependencies in sequential data through their gating mechanisms.
   - The best performing model was the GRU with units=[128], dropout=0.1, and learning rate = 0.005.

4. Visualizations:
    - Actual vs. Predicted plots demonstrated that the advanced RNNs closely tracked real stock prices with smaller deviations.

5. Performance Comparison:
    - Both models were evaluated using metrics like MSE, RMSE, and R² Value.

    - Advanced RNNs outperformed Simple RNNs by achieving better accuracy and lower prediction error across multiple target stocks. Simple RNN R2 Score = 0.8757 Advanced RNN R2 Score = 0.9947


Here is the statistics received from different Models :

|Simple RNN WO HP | AMZN   | GOOGL  | IBM      | MSFT   |
|---------------|----------|--------|----------|--------|
| MSE           | 0.0029   | 0.0003 | 0.0005   | 0.0012 |
| RMSE          | 0.0541   | 0.0164 | 0.0227   | 0.0340 |
| R² Score      | 0.8757   | 0.9805 | 0.9299   | 0.9488 |

** WO HP -- Without Hyperparamater
<br>-------------------------------------------------------------------------

|Simple RNN with HP | AMZN   | GOOGL  | IBM      | MSFT   |
|---------------|----------|--------|----------|--------|
| MSE           | 0.0009   | 0.0004 | 0.0003   | 0.0003 |
| RMSE          | 0.0295   | 0.0210 | 0.0168  | 0.0166 |
| R² Score      | 0.9630   | 0.9681 | 0.9618   | 0.9878 |

** HP -- Hyperparamater
<br>-------------------------------------------------------------------------

|Adv RNN with HP | AMZN   | GOOGL  | IBM      | MSFT   |
|---------------|----------|--------|----------|--------|
| MSE           | 0.0001   | 0.0001 | 0.0002   | 0.0003 |
| RMSE          | 0.0112   | 0.0109 | 0.0128  | 0.0159 |
| R² Score      | 0.9947   | 0.9914 | 0.9776   | 0.9888 |

** HP -- Hyperparamater

<br>
<b>Key Insights :</b>

LSTM or GRU is suitable for financial time series data, as it retains memory over long sequences and can learn non-linear relationships in time-dependent data. Normalization of features before feeding into the network is essential to improve training stability and convergence.The choice of timesteps (look-back period) had a significant effect. Shorter windows led to less accurate forecasting, while moderately sized windows gave better generalization.

