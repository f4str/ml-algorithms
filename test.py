import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from linear_regression import LinearRegression
from ridge_regression import RidgeRegression
from logistic_regression import LogisticRegression

if __name__ == '__main__':
	boston_X, boston_y = load_boston(return_X_y=True)
	cancer_X, cancer_y = load_breast_cancer(return_X_y=True)
	cancer_X = cancer_X.astype(np.float128)
	
	print('Linear Regression Test')
	regressor1 = LinearRegression()
	regressor1.fit(boston_X, boston_y)
	print('intercept:', regressor1.intercept)
	print('coefficients:', regressor1.coef)
	print('R^2:', regressor1.score(boston_X, boston_y))
	print()
	
	print('Ridge Regression Test')
	regressor2 = RidgeRegression()
	regressor2.fit(boston_X, boston_y)
	print('intercept:', regressor2.intercept)
	print('coefficients:', regressor2.coef)
	print('R^2:', regressor2.score(boston_X, boston_y))
	print()
	
	print('Logistic Regression Test')
	classifier1 = LogisticRegression()
	classifier1.fit(cancer_X, cancer_y)
	print('intercept:', classifier1.intercept)
	print('coefficients:', classifier1.coef)
	print('accuracy:', classifier1.score(cancer_X, cancer_y))
	print()
