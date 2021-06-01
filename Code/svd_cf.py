import numpy as np


def svd_collaborative_filtering(train, train_matrix, latent_variable=3,
                                learning_rate=0.1, maximum_epoches=100, regularization=0.005):
    # below code is an implement of funk-svd, following the https://github.com/gbolmier/funk-svd/README.md
    # 'Funk SVD for recommendation in a nutshell' chapter

    R = train_matrix
    # shape = num_user * num_items
    num_user = train_matrix.shape[0]
    num_item = train_matrix.shape[1]
    P = np.zeros((num_user, latent_variable))
    Q = np.zeros((latent_variable, num_item))
    # rating is predict as r_average + user_base[user] + item_base[item] + P[user,:] @ Q[:,item]
    # what we need to do is finding the optimal patameters

    r_average = np.mean(R[R != 0])
    user_base = np.zeros(num_user)
    for i in range(num_user):
        temp = R[i, :]
        if np.sum(temp != 0):
            user_base[i] = np.mean(temp[temp != 0]) - r_average
    item_base = np.zeros(num_item)
    for i in range(num_item):
        temp = R[:, i]
        if np.sum(temp != 0):
            item_base[i] = np.mean(temp[temp != 0]) - r_average
    # there are other ways to initialize user_base, item_base and P,Q

    last_loss = np.inf
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
        '''
        the code below are low
        
        '''
        R_predict = P @ Q + r_average
        R_predict = R_predict + user_base[:, np.newaxis] + item_base[np.newaxis, :]
        loss = RMSE_loss(R_predict, train)
        if loss < last_loss:
            print("epoch = " + str(epoch+1))
            print("MSE loss is equal to " + str(loss))
            last_loss = loss
        else:
            print("early stop at epoch" + str(epoch+1) + ", because the loss increased.")
            break

    R_predict = P @ Q + r_average
    R_predict = R_predict + user_base[:, np.newaxis] + item_base[np.newaxis, :]
    return R_predict


def RMSE_loss(predict_matrix, test):
    error = 0
    n = test.shape[0]
    for i in range(n):
        error += (test[i, 2] - predict_matrix[test[i, 0], test[i, 1]]) ** 2
    error = np.sqrt(error / n)
    return error
