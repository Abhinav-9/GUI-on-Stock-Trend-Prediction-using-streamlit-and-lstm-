# GUI-on-Stock-Trend-Prediction-using-streamlit-and-lstm-
Abstract: Stock market investment is one of the most complex and sophisticated way to do business. Stock market is very uncertain as the prices of stocks keep fluctuating because of several factors that makes prediction of stocks a difficult and extremely complicated task. Nowadays investors need fast and accurate information to make effective decisions and highly interested in the research area with exponentially growing technological advances of stock price prediction. Understanding the pattern of stock price of a particular company by predicting their future development and financial growth will be highly beneficial. This paper focuses on the usage of a type of recurrent neural network (RNN) based Machine learning which is known as Long Short Term Memory (LSTM) to predict stock values.

PROPOSED SYSTEM
 We propose to use LSTM (Long Short Term Memory) algorithm to provide efficient stock price prediction. 
LSTM –an overview
 
  A special type of RNN, which can learn long-term dependence, is called Long-Short Term Memory (LSTM). LSTM enables RNN to remember long-term inputs. Contains information in memory, similar to computer memory. It is able to read, write and delete information in its memory. This memory can be seen as a closed cell, with a closed description, the cell decides to store or delete information. In LSTM, there are three gates: input, forget and exit gate. These gates determine whether new input (input gate) should be allowed, data deleted because it is not important (forget gate), or allow it to affect output at current timeline (output gate) [2]. 
1. Forget gate: The forget gateway determines when certain parts of the cell will be inserted with information that is more recent. It subtracts almost 1 in parts of the cell state to be kept, and zero in values to be ignored. 
2. Input gate : Based on the input (e.g., previous output o (t-1), input x (t), and the previous state of cell c (t-1)), this network category reads the conditions under which any information should be stored (or updated) in the state cell. 
3. Output gate: Depending on the input mode and the cell, this component determines which information is forwarded in the next location in the network.

SYSTEM ARCHITECTURE
 

Obtaining dataset and pre-processing 
The obtained data contained five features:
 1. Date: Date of stock price.
 2. Opening price: When trading begins each day this is opening price of stock. 
3. High: The highest price at which the stock was traded during a period(day). 
4. Low: The Lowest price at which the stock was traded during a period(day).
 5. Volume: How much of a given financial asset has traded in a period of time. 
6. Close Interest: The last price at which a particular stock traded for the trading session. 
 
 Stock market information is available from key sources: Tiingo API, Yahoo and Google Finance. These websites give APIs from which stock dataset can be obtained from various companies by simply specifying parameters.
 The data is processed into a format suitable to use with prediction model by performing the following steps:
 1. Transformation oftime-seriesdata intoinput-output components for supervised learning. 
2. Scaling the data to the [-1, +1]range.

RESULT ANALYSIS 
Actual price and closing price of Apple company, a large stock. The model was trained in bulk sizes of 321and 200 epochs, and the forecasts were made very similar to stock prices, as seen in the graph.
 
 
  



 CONCLUSION 
Stock investing has attracted the interest of many investors around the world. However, making a decision is a difficult task as many things are involved. By investing successfully, investors are eager to predict the future of the stock market. Even the slightest improvement in performance can be enormous. A good forecasting system will help investors make investments more accurate and more profitable by providing supporting information such as future stock price guidance. In addition to historical prices, other related information could affect prices such as politics, economic growth, financial matters and the atmosphere on social media. Numerous studies have proven that emotional analysis has a significant impact on future prices. Therefore, the combination of technical and basic analysis can produce very good predictions.
REFERENCES 
[1] Kim, S., Ku, S., Chang, W., & Song, J. W. (2020). Predicting the Direction of US Stock Prices Using Effective Transfer Entropy and Machine Learning Techniques. IEEE Access, 8, 111660-111682.
 [2] Nandakumar, R., Uttamraj, K. R., Vishal, R., & Lokeswari, Y. V. (2018). Stock price prediction using long short term memory. International Research Journal of Engineering and Technology, 5(03). 
[3] Roondiwala, Murtaza, Harshal Patel, and Shraddha Varma. "Predicting stock prices using LSTM." International Journal of Science and Research (IJSR) 6.4 (2017): 1754-1756.
