# SpamClassification Based on Email Corpus

* Written in MATLAB and R

I used word frequencies in each email to predict whether an email is spam or not. Since there were 4000 emails and 1899 words in the vocabulary list, it was a typical high dimensional binary classification problem. Mainly I tried two kinds of methods, SVM and logistic regression with elastic net penalization. Finally conducted an ensemble model averaging two methods will lead to a better result.

First, I did a preprocessing with the email corpus. Removing HTML tags, spotting numbers, email address, URLs in the email content, then by using porter stemming method, essential words were extracted from the emails. 

Then I built classification models. The predictors are the term frequencies for 1899 terms, the response is a binary variable whether it’s spam or not. Thus, there will be 1899 variables and 4000 observations in the training data set. In the text classification problem with large number of variables and few observations, it’s suitable for logistic regression and SVM with linear kernel. For SVM model, since this risk overfitting in a high dimensional feature-space, so I choose linear kernel, using 10-fold cross-validation to choose the best cost parameter. In logistic regression with elastic net regulation, I first choose AUC as measurement to choose the cost parameters which maximize AUC. Then another cross-validation to choose the best probability threshold. 

At last, I averaged top three models to build a final ensemble model. The accuracy is 98.9%, F1 score is 0.992. It predicted very well and overcome the problem of unbalance in the training data set.
