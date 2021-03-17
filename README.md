# Sentiment Analysis for Hotel reviews
Developed here is a text classification model for sentiment analysis of hotel reviews. This python code comprises of two parts:

#### Part 1:
Reading the reviews and extracting the feature vectors with help of the rules given in section 5.1.1 of this pdf: https://web.stanford.edu/~jurafsky/slp3/5.pdf

#### Part 2:
Implementation of logistic regression and training the model using the feature vectors produced in part 1. The weights are corrected with Stochastic Gradient Descent Algorithm using Cross Entropy as the loss function. 

The model produced an accuracy of 92.1% on the test set of the data (the split of the data was 80%-20% with 20% of the data used for testing).
