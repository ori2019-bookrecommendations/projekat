import sys
from time import time
from BookRecommendation import utils
from BookRecommendation.recommendation_algorithms import collaborativeFiltering, contentBasedFiltering

if __name__ == "__main__":
    argv = sys.argv[1:]

    model = None
    try:
        model = argv[0]
    except IndexError:
        sys.exit("Please specify model to train: `cbf`, `svd`, `neumf`")

    books, book_tags, ratings, tags, to_read, users = utils.load_data()
    test, train = utils.load_test_and_train_data()

    if model == 'cbf':
        start = time()
        print("CBF training started")
        cbf = contentBasedFiltering.ContentBasedFiltering()
        cbf.train()
        end = time()
        print("CBF training finished, saving to /inputs/cbf/cbf")
        print("Training time: " + str(end - start))
        cbf.save('./inputs/cbf/cbf2')
    elif model == 'svd':
        svd = collaborativeFiltering.SVDCollaborativeFiltering(ratings)
        start = time()
        print("SVD training started")
        svd.train_model()
        end = time()
        print("SVD training finished, saving to /inputs/cf_models/svd-model.cfm")
        print("Training time: " + str(end - start))
        svd = svd.save('./inputs/cf_models/svd-model.cfm')
        print("SVD model evaluation")
        svd_mae, svd_rmse = svd.test_model()
        print("SVD MAE: " + str(svd_mae))
        print("SVD RMSE: " + str(svd_rmse))
    elif model == 'neumf':
        try:
            num_latent_factors_user = int(argv[1])
            num_latent_factors_book = int(argv[2])
            num_latent_factors_matrix_factorization = int(argv[3])
            batch_size = int(argv[4])
            epochs = int(argv[5])
        except (IndexError, ValueError):
            sys.exit("Please specify number of latent factors for the user, book, matrix factorization,"
                     " batch size and number of epochs.\n"
                     "Correct call example > neumf 20 30 6 131072 25")

        ncf = collaborativeFiltering.NeuralCollaborativeFiltering(
            ratings, num_latent_factors_user, num_latent_factors_book, num_latent_factors_matrix_factorization)

        start = time()
        print("NeuMF training started")
        ncf.train_model(train, batch_size, epochs)
        end = time()
        print("NeuMF training finished, saving to /inputs/cf_models/svd-model.cfm")
        print("Training time: " + str(end - start))
        ncf.save("./inputs/cf_models/ncf-model2.cfm")
        print("NeuMF model evaluation")
        ncf_mae, ncf_rmse = ncf.test_model(test, batch_size)
        print("NeuMF MAE: " + str(ncf_mae))
        print("NeuMF RMSE: " + str(ncf_rmse))
    else:
        sys.exit("Please specify model to train: `cb`, `svd`, `neumf`")
