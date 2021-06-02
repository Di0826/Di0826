import numpy as np

def svd_collaborative_filtering(train, user_data, item_data, latent_variable=8,
                                learning_rate=0.01, maximum_epoches=25, regularization=0.005):
    # below code is an implement of funk-svd, following the https://github.com/gbolmier/funk-svd/README.md
    # 'Funk SVD for recommendation in a nutshell' chapter

    train_matrix = np.zeros((user_data.shape[0], item_data.shape[0]))
    # shape = num_user * num_items
    train_matrix[train[:, 0], train[:, 1]] = train[:, 2]
    num_user = train_matrix.shape[0]
    num_item = train_matrix.shape[1]
    P = np.zeros((num_user, latent_variable))
    Q = np.zeros((latent_variable, num_item))
    # rating is predict as r_average + user_base[user] + item_base[item] + P[user,:] @ Q[:,item]
    # what we need to do is finding the optimal patameters

    r_average = np.mean(train_matrix[train_matrix != 0])
    user_base = np.zeros(num_user)
    item_base = np.zeros(num_item)
    # there are other ways to initialize user_base, item_base and P,Q

    last_loss = np.inf
    print('----------svd_collaborative_filtering start training----------')
    for epoch in range(maximum_epoches):
        for rating_idx in range(train.shape[0]):
            user = int(train[rating_idx, 0])
            item = int(train[rating_idx, 1])
            rating = train[rating_idx, 2]
            # predict:
            dot_p_q = 0
            # dot product of P[u,:] and Q[:,i],
            for variable_idx in range(latent_variable):
                dot_p_q += P[user, variable_idx] * Q[variable_idx, item]
            error = rating - (r_average + user_base[user] + item_base[item] + dot_p_q)

            # update :
            user_base[user] += learning_rate * (error - regularization * user_base[user])
            item_base[item] += learning_rate * (error - regularization * item_base[item])
            for variable_idx in range(latent_variable):
                P[user, variable_idx] += learning_rate * (error * Q[variable_idx, item]
                                                          - regularization * P[user, variable_idx])
                Q[variable_idx, item] += learning_rate * (error * P[user, variable_idx]
                                                          - regularization * Q[variable_idx, item])
        if (epoch + 1) % 5 == 0:
            predict_matrix = P @ Q + r_average
            predict_matrix = predict_matrix + user_base[:, np.newaxis] + item_base[np.newaxis, :]
            loss = RMSE_loss(predict_matrix, train)
            if loss < last_loss:
                print('At epoch [%d/%d] RMSEloss: %.5f' % (epoch + 1, maximum_epoches, loss))
                last_loss = loss
            else:
                print("Early stop at epoch" + str(epoch+1) + ", because the loss increased.")
                break

    predict_matrix = P @ Q + r_average
    predict_matrix = predict_matrix + user_base[:, np.newaxis] + item_base[np.newaxis, :]
    return predict_matrix


def RMSE_loss(predict_matrix, test):
    predict = predict_matrix[test[:, 0], test[:, 1]]
    MSE = np.sum(np.square(predict - test[:, 2]))/test.shape[0]
    RMSE = np.sqrt(MSE)
    return RMSE
