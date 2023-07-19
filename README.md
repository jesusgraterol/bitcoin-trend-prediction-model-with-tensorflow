# Bitcoin Trend Prediction Model (TensorFlow)

In this notebook, we build and train a Multi-Step / Multi-Output Regression Model powered by TensorFlow & Keras in order to predict Bitcoin's future trend. 

Even though a hyperparameter tuning mechanism is out of the scope of this notebook, we will be trying a series of networks such as: Dense, Convolutional with and without Max Pooling, Long Short-Term Memory (LSTM), Gated Recurrent Unit, etc.

The metrics we will be using to evaluate how well our model generalizes when predicting on unseen data are MAE (Mean Absolute Error) and MSE (Mean Squared Error). The Neural Network that scores the lowest loss is the one that should be deployed to production.

It is also important to mention that this model will be trained based on the moving averages as the Bitcoin's price is full of noise, and we don't want our model to learn useless patterns.

[View Notebook](./bitcoin-trend-prediction-model-tensorflow.ipynb)
