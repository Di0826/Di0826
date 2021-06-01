import numpy as np
import data_reader
import svd_cf
user_data = data_reader.read_user()
item_data = data_reader.read_item()
latent_variable_SVD = 8  # used by svd
learning_rate_SVD = 0.01
maximum_epoches = 25
regularization = 0.005

for test_num in range(1, 2):
    print("Test" + str(test_num) + ":")

    train = data_reader.read_rating_train(test_num)
    test = data_reader.read_rating_test(test_num)

    train_matrix = np.zeros((user_data.shape[0], item_data.shape[0]))
    # shape = num_user * num_items
    train_matrix[train[:, 0], train[:, 1]] = train[:, 2]
    predict_matrix = svd_cf.svd_collaborative_filtering(train, train_matrix, latent_variable_SVD,
                                                        learning_rate_SVD, maximum_epoches, regularization)
    # using svd to fill the matrix
    print("RMSE loss of " + "test " + str(test_num) + " is equal to " + str(svd_cf.RMSE_loss(predict_matrix, test)))


