import numpy as np
import torch


def feature_assembler(data, user_data, item_data):
    num_row = data.shape[0]
    num_column = user_data.shape[1] + item_data.shape[1]
    feature = np.zeros((num_row, num_column))
    user_index = data[:, 0]
    feature[:, 0:user_data.shape[1]] = user_data[user_index, :]
    item_index = data[:, 1]
    feature[:, user_data.shape[1]:] = item_data[item_index, :]
    return feature


def neuralnet_predictor(train, user_data, item_data):
    torch.manual_seed(1)
    np.random.seed(1)

    features = torch.tensor(feature_assembler(train, user_data, item_data), dtype=torch.float)
    ratings = torch.tensor(train[:, -2], dtype=torch.float)

    ratings = ratings.unsqueeze(1)
    input_neurons = user_data.shape[1] + item_data.shape[1]
    hidden_neurons = 10
    output_neurons = 1
    learning_rate = 0.01
    num_epoch = 100
    net = torch.nn.Sequential(
        torch.nn.Linear(input_neurons, hidden_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_neurons, hidden_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_neurons, output_neurons)
    )
    loss_func = torch.nn.MSELoss()
    optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate)
    last_loss = np.inf
    print('----------neuralnet_predictor start training----------')

    for epoch in range(num_epoch):
        pred = net(features)
        loss = loss_func(ratings, pred)

        if (epoch + 1) % 5 == 0:
            print('At epoch [%d/%d] RMSEloss: %.6f' % (epoch + 1, num_epoch, np.sqrt(loss.item())))
            if epoch > 10 and last_loss - loss.item() < 1e-6:
                print('Stop training since the delta of loss is too small')
                break
            else:
                last_loss = loss.item()
        net.zero_grad()
        loss.backward()
        optimiser.step()

    # calculate predict_matrix
    user_idx, item_idx = np.meshgrid(np.arange(user_data.shape[0]), np.arange(item_data.shape[0]))
    user_idx = user_idx.flatten()
    item_idx = user_idx.flatten()
    features2 = feature_assembler(np.concatenate((user_idx.reshape(-1, 1), item_idx.reshape(-1, 1)), axis=1), user_data, item_data)
    pred = net(torch.tensor(features2, dtype=torch.float))
    pred = pred.detach().numpy()
    predict_matrix = np.transpose(pred.reshape(item_data.shape[0], user_data.shape[0]))

    return predict_matrix


