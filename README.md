# FraudDetectionCaseStudy

## Overview

This was a collaboration with Richard Chan and Ethan Lemus.

The goal of this case study was to create a complete fraud detection application that could interpret, display, and store fraud predictions. For each of these smaller pieces there was a respective architechture that was built namely an accurate machine learning model, a web application, and a database system. The web application was created using Flask and ran on an AWS EC2 instance. The database was constructed using MongoDB. The sections below intail how the model was developed.

## Process Flow

First we identified features that had value in modeling Then preprocessed those features. A wide array of models(Gradient Boosted, Random Forest, Neural Network, etc.) were used with varing hyperprameters and the presence of SMOTE. These models were validated using a test split of 20% on their ROC curves.

### Preprocessing
The target of our model are the account types containing the word 'fraud' in them. We created a boolean column that indicated if that row was fraudulent then discarded the account type column. 

Features were broken into their respective anticipated types (Boolean, Categorical, Date, and Continuous) and converted to those types. The Nans found in the continuous and date variables were converted to the median of thier respective columns. 

The date columns were converted from Unix timestamps to datetime objects. We extracted the hour and the day of week into new columns then discarded the rest.

There were 8 features that we deemed had no relevant information we could easily access so we discarded them.

There were 3 features that we felt had not internal relevance other than their presence. We discarded these columns and replaced them with booleans if they previously existed or not.

The accumulation of the quantity sold was extracted from the 'ticket_types' column. The number of previous payouts was extracted from the 'previous_payouts' column.

### Accuracy Metrics
ROC AUC (Area Under the Receiver Operating Characteristic Curve) was chosen as our evaluation metric as in this case study, since accuracy score does not reflect well on our model score as the data is imbalanced with only a minority of the data classified as fraud. The model can still return a high accuracy score even if it only predicts false for every single item.

Thus, we are more interested in both the false positive rate (precision) and true positive rate (recall), which is simulataneously captured in the AUC score versus only recall and precision scores.

### Validation and Testing

Validation was conducted by comparing our models prediction of the test data subset(20% of total) to the actual test data.

### Parameter Tuning

Using GridSearchCV from sklearn, we input a range of values on a couple of parameters that are usually hypertuned for random forest and gradient boosted models. We then choose best parameters chosen by GridSearchCV.

##Results

The model that was the most accurate was a Gradient Boosted model trained with SMOTE oversampling. The accuracy was 98.7% with a precision of 95.4% and recall of 91.2% at a threshold of 50%.

The model was deployed in a Flask app running in an AWS EC2 instance. The application displayed the probability of fraud in live transactions these predictions were then stored in MongoDB. 

## Future Steps

Further feature extraction could increase our model accuracy. Such features could be the cost in the 'ticket_types' column or to run an NLP evaluation on any of the text features.

