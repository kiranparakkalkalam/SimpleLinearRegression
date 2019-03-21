'''
This is a simple linear regression implementation based on Gradient Descent.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict_price(x, theta):
    ''' Vectorized implementation of calculating the hypothesis'''
    return np.dot(x, theta)

def calculate_cost(theta, x, y):
   prediction = predict_price(x, theta)
   return ((prediction - y)**2).mean() / 2

def plot_line(data, theta, x, y):
    '''Plot the values and the best fit line'''
    predicted_values_y = predict_price(x, theta)
    plt.figure(figsize=(16, 8))
    plt.scatter(data['TV'], data['sales'], c="black")
    plt.plot(data['TV'], predicted_values_y, c="blue", linewidth=2)
    plt.xlabel("Money spent on TV ads ($)")
    plt.ylabel("Sales($)")
    plt.show()

def gradient_descent_linear_regression(filename, alpha=0.05, iteration=100):
    ''' Using the gradient descent to find the theta's and cost'''
    data = pd.read_csv(filename)
    data.drop(['Unnamed: 0'], axis=1)
    x = np.column_stack((np.ones(len(data['TV'])), data['TV']))
    y = data['sales']
    theta = np.zeros(2)
    theta0 = []
    theta1 = []
    costs = []
    for i in range(iteration):
        predicted_price = predict_price(x, theta)
        t0 = theta[0] - alpha*(predicted_price - y).mean()
        t1 = theta[1] - alpha*((predicted_price - y)*x[:, 1]).mean()
        theta = np.array([t0, t1])
        # To keep track of the cost
        cost = calculate_cost(theta, x, y)
        theta0.append(t0)
        theta1.append(t1)
        costs.append(cost)
        if i % 10 == 0:
            print(f"Iteration: {i+1},Cost = {cost},theta = {theta}")
            plot_line(data, theta, x, y)

if __name__ == "__main__":
    gradient_descent_linear_regression("data/Advertising.csv")
