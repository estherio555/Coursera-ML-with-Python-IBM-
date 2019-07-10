# Projects - Predicting Loan payment Status
In this project, I completed a notebook where I built a classifier to predict whether a loan case will be paid off or not.

I used the training set to build various model, then use the test set to report the accuracy of the model.
I used the following algorithm:

K Nearest Neighbor(KNN)

Decision Tree

Support Vector Machine

Logistic Regression

The results is reported as the accuracy of each classifier, using the following metrics when these are applicable:

Jaccard index

F1-score

LogLoss

### About dataset

This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:

| Field          | Description                                                                           |
|----------------|---------------------------------------------------------------------------------------|
| Loan_status    | Whether a loan is paid off on in collection                                           |
| Principal      | Basic principal loan amount at the                                                    |
| Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
| Effective_date | When the loan got originated and took effects                                         |
| Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
| Age            | Age of applicant                                                                      |
| Education      | Education of applicant                                                                |
| Gender         | The gender of applicant                                                               |
