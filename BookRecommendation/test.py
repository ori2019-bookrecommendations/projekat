from BookRecommendation.recommendation_algorithms import collaborativeFiltering
import sys
import pandas as pd

# test.py $batch-size
if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) != 1:
        sys.exit("Expected one argument for batch size!")

    try:
        batch_size = int(argv[0])
    except ValueError:
        sys.exit("Invalid argument %s, expected int!" % argv[0])

    ratings = pd.read_csv('./inputs/ratings.csv')

    ncf = collaborativeFiltering.NeuralCollaborativeFiltering(ratings, 20, 30, 6)
    svd = collaborativeFiltering.SVDCollaborativeFiltering(ratings)

    ncf.load('./inputs/cf_models/ncf-model.cfm')
    svd = svd.load('./inputs/cf_models/svd-model.cfm')
    test = pd.read_csv('./outputs/test_set.csv')

    # Tests the models
    ncf_mae, ncf_rmse = ncf.test_model(test, batch_size)
    print("NeuMF MAE: " + str(ncf_mae))
    print("NeuMF RMSE: " + str(ncf_rmse))

    svd_mae, svd_rmse = svd.test_model()
    print("SVD MAE: " + str(svd_mae))
    print("SVD RMSE: " + str(svd_rmse))
