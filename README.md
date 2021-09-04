## I) Abstract
In this paper, I have built a multivariate analysis model to predict Ethereum price movement. Crypto prices depends on various factors and their complex dynamics which makes them a difficult problem in real world. The purpose of this paper is to analyses the capability of a neural network to solve this problem efficiently. Recurrent Neural Networks (RNN) has demonstrated its capability of addressing complex time series problems.


## II) Introduction

Crypto price prediction is one of the most important business problems that has attracted the interest of all the stakeholders. To improve the performance, reliability of forecasting and the complexity of algorithms used in the process of solving this problem. However, the methods I have found yet are either based on simple linear regression assumptions or do not make full use of the data available and only consider one factor while forecasting (non-linear univariate models and deep learning models). Some researchers have also tried a combination of ANN to use human like reasoning for this problem. But the prediction is still open. The asset prices are highly dynamic and have non-linear relationships and is dependent on many factors at the same time. I have tried to solve this problem of the market forecasting using multivariate analysis.

I have used multivariate RNN. It is proven that deep learning algorithms have the ability to identify existing patterns in the data and exploiting them by using a soft learning process. Unlike other statistical and machine learning algorithms, deep learning architectures are capable to find short term as well as long term dependencies in the data and give good predictions by finding these hidden relationships.

I have proposed a 3-level methodology of our work. First, I preprocessed the data to make the data multidimensional and suitable for our network architectures. Next, I splitted the data into train, validation and test sets and train our models on the training data. At the final step, I made predictions using the models trained in the previous step on test data and calculated and analyzed various error matrices. 

## III) Methodology

- **Raw Data:**
    
    We used the historical Ethereum prices of Binance 
    obtained from www.cryptodatadownload.com . It contains 5670 records of daily
    stock prices of the stocks from 09/08/1996 to 22/02/2019. Each record
    contains information of opening, closing, high, and low value of ETH
    as well as the volume of the stock sold on that day.

- **Data Pre-processing:**
    
    First, we remove some redundant and noisy data, such as the records
    with volume 0 and the records that are identical to previous record. For
    unifying the data range, we applied normalization on volume and
    mapped the values to a range of 0 to 1.  
    
- **Testing and Error Calculation:**

    Each model has been tested on the test set and their Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R2-score are calculated.



### Model: Multivariate-RNN:

The model is trained on the series of records containing High price (Highest Correlation with target), Volume (Lowest Correlation with
target) and Close price of the stock. Different parameters of this ANN are as follows:

 - Timesteps: 72 (3 days)
 - Neurons in each Layer: 50 and 45
 - Learning Rate: 0.001
 - Batch Size: 32
 - Total Trainable Parameters: 7227

The training data is fed to this network and the model is trained for 200 epochs on the training data and validated by the validation data.

  

## IV) Tools and Technology Used

I used Python syntax for this project. As a framework I used
Keras, which is a high-level neural network API written in Python. But
Keras can’t work by itself, it needs a backend for low-level operations.
Thus, we installed a dedicated software library — Google’s TensorFlow.

For scientific computation, we installed Scipy. As a development environment I used Google Colab.
I used Matplotlib for data visualization, Numpy for various array
operations and Pandas for data analysis.

## V) Results

** Table 1: Results of model on test data **

| Model | Features Used | MSE | RMSE | R2-score |
|---|---|---|---|---|
| Multivariate-RNN | [Open,High,Low,Close,Volume(N)] | 0.0002176880 | 0.0139925408 | 0.9423308750 |


## VI) References

  [1] K. Soman, V. Sureshkumar, V. T. N. Pedamallu, S. A. Jami, N. C.
  Vasireddy and V. K. Menon, “Bulk price forecasting using spark over
  nse data set,” Springer, 2016, pp. 137–146. International Conference
  on Data Mining and Big Data.

  [2] C. S. Lin, H. A. Khan and C. C. Huang, 'Can the neuro fuzzy model
  predict stock indexes better than its rivals?', Proc. CIRJE, CIRJE-F-
  165, Aug, 2002.

  [3] Z.P. Zhang, G.Z. Liu, and Y.W. Yang, “Stock market trend
  prediction based on neural networks, multiresolution analysis and
  dynamical reconstruction,” pp.155-56, March 2000. IEEE/IAFE
  Conference on Computational Intelligence for Financial Engineering,
  Proceedings (CIFEr).
  
  [4] Yoshua Bengio, I. J. Goodfellow, and A. Courville, “Deep
  learning,” pp. 436–444, Nature, vol. 521, 2015.

  [5] Razvan Pascanu, Tomas Mikolov, Yoshua Bengio, “On the
  difficulty of training Recurrent Neural Networks”, arXiv:1211.5063.

  [6] Kyunghyun Cho (2014). "Learning Phrase Representations using
  RNN Encoder-Decoder for Statistical Machine Translation".
  arXiv:1406.1078.

  [7] https://www.cryptodatadownload.com/data/binance/



