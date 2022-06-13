import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.optim as optim

features = pd.read_csv('temps.csv')
# print('features.shape:', features.shape)


years = features['year']
months = features['month']
days = features['day']

# transform to datetime format
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day))
         for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]


# Plot graph
def graph():
    # Style
    plt.style.use('fivethirtyeight')

    # set layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, figsize=(10, 10))
    fig.autofmt_xdate(rotation=45)

    # Actual max temperature today
    ax1.plot(dates, features['actual'])
    ax1.set_xlabel('')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Max Temp')

    # plot yesterday data
    ax2.plot(dates, features['temp_1'])
    ax2.set_xlabel('')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Max Temp of yesterday')

    # the day before yesterday
    ax3.plot(dates, features['temp_2'])
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Max Temp of day before yesterday')

    # friends estimation
    ax4.plot(dates, features['friend'])
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Temperature')
    ax4.set_title('Friend Estimate')

    plt.tight_layout(pad=2)
    # plt.show()


# Process Data
# hot encoding
features = pd.get_dummies(features)

# actual data
labels = np.array(features['actual'])

# drop the actual data off the features DataFrame
features = features.drop('actual', axis=1)

# save header names in a list
feature_list = list(features.columns)

# transform to an appropriate format
features = np.array(features)

input_features = preprocessing.StandardScaler().fit_transform(features)

'''
# 1st way (complex way) to build a NN model

x = torch.tensor(input_features, dtype=float)
y = torch.tensor(labels, dtype=float)

# Initialize all parameters
weights = torch.randn((14, 128), dtype=float, requires_grad=True)
biases = torch.randn(128, dtype=float, requires_grad=True)
weights2 = torch.randn((128, 1), dtype=float, requires_grad=True)
biases2 = torch.randn(1, dtype=float, requires_grad=True)

learning_rate = 0.001
losses = []

for i in range(1000):
    # calculate hidden layers
    hidden = x.mm(weights) + biases
    # add an activation function
    hidden = torch.relu(hidden)
    # outcome layer (predict results)
    predictions = hidden.mm(weights2) + biases2
    # calculate loss
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())

    # print the value of loss 10 times in 1000 echo
    if i % 100 == 0:
        print('loss:', loss)

    # backtrack calculate
    loss.backward()

    # update all parameters
    weights.data.add_(- learning_rate * weights.grad.data)
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)

    # clear all parameters to zero after each iteration
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()
'''

# 2nd way (simple way) to build a NN model
input_size = input_features.shape[1]
print(input_features.shape)
hidden_size = 128
output_size = 1
batch_size = 16
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)
# Mean squared error (MSE) is the most commonly used loss function for regression.
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

# train the NN
losses = []
for i in range(1000):
    batch_loss = []
    # MINI-Batch method to train
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    # print the loss
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# Predict the training results
x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy()

# transform to datetime format
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# create a DataFrame to save dates and their corresponding labels (true data)
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})

# create a DataFrame to same dates and their corresponding predicting data
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})

# Plot the actual data and prediction data in a graph
# true data
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')

# prediction data
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60');
plt.legend()

# graph name
plt.xlabel('Date');
plt.ylabel('Maximum Temperature (F)');
plt.title('Actual and Predicted Values');

plt.show()
