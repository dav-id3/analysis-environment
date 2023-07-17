from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
import seaborn as sns
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from src.utils import NormalDistributionValidator, collinearity_check
import warnings

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


def main() -> None:
    boston = fetch_california_housing()
    y = boston.target
    normal_dist_validator_y = NormalDistributionValidator(y)
    normal_dist_validator_y.test_qqplot()
    sns.displot(y)
    plt.show()

    y_boxcox, lambda_boxcox = boxcox(y)  # boxcox transformation (Xc should follow bormal destribution)
    normal_dist_validator_boxcox = NormalDistributionValidator(y_boxcox)
    normal_dist_validator_boxcox.test_kolmogorov_smirnov()
    normal_dist_validator_boxcox.test_qqplot()
    sns.displot(y_boxcox)
    plt.show()

    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    print(df.head())

    df_std = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)  # standardization
    collinearity_check(df_std)
