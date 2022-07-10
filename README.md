# Regularized linear regression project

In this project we use a dataset with features related to socio demographic and health resources by county in the United States.

The dataset has many features. We use Lasso to select features and then we run a OLS linear regression with the selected ones.

For Lasso, firs we run one with an arbitrary alpha equal to 10. Then, we choose alpha by cross-validation.