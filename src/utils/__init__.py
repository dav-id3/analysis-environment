from typing import List, Tuple, Union
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statistics import mean, median, variance, stdev
from scipy.stats import shapiro, ks_1samp, norm, probplot, multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_measurements(target_varuables: List[float]) -> None:
    print(f"n: {len(target_varuables)}")
    print(f"mean: {mean(target_varuables)}")
    print(f"median: {median(target_varuables)}")
    print(f"variance: {variance(target_varuables)}")
    print(f"stdev: {stdev(target_varuables)}")


class NormalDistributionValidator:
    """
    Normal distribution validator
        args:
            target_varuables(List[float]): target varuables
    """

    def __init__(self, target_varuables: List[float]) -> None:
        self.target_varuables = target_varuables

    def test_shapiro_wilk(self) -> None:
        """
        Shapiro-Wilk Test to check if the distribution is normal
        to be applied to a small sample size (n <= 50)
        p < 0.05: reject the null hypothesis (not normal)
        """
        stat, p = shapiro(self.target_varuables)
        print(f"Shapiro-Wilk Test: stat={stat}, p={p}")

    def test_kolmogorov_smirnov(self) -> None:
        """
        kolmoogorov-smirnov Test to check if the distribution is normal
        to be applied to a large sample size (n > 50)
        p < 0.05: reject the null hypothesis (not normal)
        """
        stat, p = ks_1samp(self.target_varuables, norm.cdf)
        print(f"Kolmogorov-Smirnov Test: stat={stat}, p={p}")

    def test_qqplot(self) -> None:
        """
        QQ plot to check if the distribution is normal
        """
        probplot(self.target_varuables, dist="norm", plot=plt)


def collinearity_check(Xc_df: pd.DataFrame, *, model=None, alpha: float = 1.0, emph: bool = True) -> None:
    """
    check collinearity of explanatory variables
    R2 > 0.9: high collinearity
    R2 > 0.8: moderate collinearity
    R2 > 0.7: weak collinearity

    Parameters
    ----------
    Xc : pd.DataFrame(->np.ndarray(m, n))
        Input data (each data stored in a row). It should be standardized beforehands.
    model
        Regression model (default Ridge).
    alpha : float
        Hyper parameter of Ridge (default 1.0),
        ignored if model is not None.
    emph : bool
        Emphasize the result by R2 score or not.

    Returns
    -------
    rc : np.ndarray(n, n)
        Regression coefficient, emphasized by R2 score if emph is True.
    scores : np.ndarray(n)
        R2 scores.
    """

    Xc = Xc_df.to_numpy(copy=True)
    header = Xc_df.columns

    if model is None:
        model = Ridge(alpha=alpha)

    m, n = Xc.shape
    if n < 2:
        raise ValueError()

    # 戻り値
    rc = np.empty((n, n))  # 回帰係数(regression coefficient)
    r2_scores = np.empty(n)  # R2スコア(決定係数)

    X = np.copy(Xc)
    for i in range(n):
        y = np.copy(X[:, i])  # 自分を他の変数で回帰させる
        X[:, i] = 1  # 自分は多重共線性の定数項に対応させる

        model.fit(X, y)
        y_calc = model.predict(X)

        score = r2_score(y, y_calc)
        if score < 0:
            # R2 スコアが 0 以下なら線形性なしとみなす
            r2_scores[i] = 0
            rc[i] = 0
        else:
            r2_scores[i] = score
            if emph:
                # 係数が大きくても R2 スコアが 0 に近ければ 0 になるように加工
                rc[i] = model.coef_ * score
            else:
                rc[i] = model.coef_

        X[:, i] = y

    for i in range(n):
        print(f"{i}({header[i]}): rc values={rc[i]}, r2 value={r2_scores[i]}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(rc), vmin=-1, vmax=1, cmap="bwr", cbar=True)
    plt.ylabel("target variable", fontsize=16)
    plt.xlabel("used variables to explain the target", fontsize=16)
    plt.show()


class BeyesLinearRegression:
    """
    ベイズ線形回帰(パラメータの事前分布が多変量正規分布N(0, alpha * I)に従う場合)
    以下のモデルを仮定し、基底関数Φの係数ベクトルwを推定する.
    y = ⟨w, Φ(x)⟩ + ε ただしε ~ N(0, σ^2)
    前提条件:
        1. 目的変数yは平均⟨w, Φ(x)⟩、分散σ^2の正規分布に従う.yの全体Yが正弦波だとしても、各yは正規分布に従うという意味.
        3. 事前分布p(w)の平均は0、分散はα*I(αはハイパーパラメータ)に従う多変量正規分布に従うとする.
    Note:
        1. 事前分布p(w)は正規分布に従うと仮定しているので、パラメータwの推定時にはその推定値が平均値m_0から離れるほど大きなペナルティが課される.

    initialize: 初期化. 分散σ、基底関数の種類と項数M、wの事前分布のパラメータm_0, alphaをハイパーパラメータとして指定する.
        args;
            x_train(np.ndarray): train用の説明変数
            phi_type(str): 基底関数の種類. "linear_deg(1-3)" or or "sin" or "gaussian"
            sigma(float): σ
    calculate_posterior: 事後分布の期待値と分散を計算する.
        args;
            y_pd_train(pd.ndarray): 目的変数
    predict: 予測する
        args;
            x_pd_test(pd.ndarray): 説明変数
        return;
            y_pred(np.ndarray): 予測値
            S_pred(np.ndarray): 予測値の分散
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        phi_type: str = "gaussian",
        sigma: float = 0.1,
        mu_0: float = 0.0,
        alpha: float = 1.0,
        M: int = 5,
    ):
        self.x_train = x_train
        self.y_train = y_train

        self.sigma = sigma
        self.beta = 1.0 / (sigma**2)
        self.alpha = alpha

        # 基底関数の項数
        self.M = M

        # phi: yを説明するための基底関数. yがどのような基底関数の線形結合で表現されるかによって変わる.
        # 基底関数の各項の係数を求めることがベイズ線形回帰の目的となる.
        self.phi_type = phi_type
        self.phi = self.x_to_phi(x_train, self.phi_type)

        # パラメータの事前分布が正規分布N(mu_0 * 単位ベクトル, alpha * I)に従うとして初期化.この場合結果はリッジ回帰と同じ意味を持つ.
        input_dim = self.phi.shape[1]
        self.mu = np.ones(input_dim) * mu_0  # パラメータwの事前分布の平均値を0とする.
        self.S = np.identity(input_dim) * alpha  # パラメータwの事前分布の分散共分散行列を単位行列とする.

    def x_to_phi(self, x: np.ndarray, phi_type: str) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if phi_type == "linear_deg1":
            # 基底関数は1次多項式. 切片を含めるために1を追加
            return np.concatenate([np.ones(x.shape[0]).reshape(-1, 1), x], axis=1)
        elif phi_type == "linear_deg2":
            # 基底関数は2次多項式. 切片を含めるために1を追加
            return np.concatenate([np.ones(x.shape[0]).reshape(-1, 1), x, x**2], axis=1)
        elif phi_type == "linear_deg3":
            # 基底関数は3次多項式. 切片を含めるために1を追加
            return np.concatenate([np.ones(x.shape[0]).reshape(-1, 1), x, x**2, x**3], axis=1)
        elif phi_type == "sin":
            # 基底関数はsin関数のM次関数.
            return np.concatenate([np.sin(2 * np.pi * x * m) for m in range(0, self.M)], axis=1)
        elif phi_type == "gaussian":
            x_range_min = 0.0
            x_range_max = 1.0
            # x_range_max = np.max(x)
            # x_range_min = np.min(x)
            return np.concatenate(
                [
                    np.exp(-((x - m) ** 2) / (2 * 0.1**2))
                    for m in np.arange(x_range_min, x_range_max + 0.01, (x_range_max - x_range_min) / (self.M - 1))
                ],
                axis=1,
            )
        else:
            raise ValueError("phi_type must be 'linear_deg(1-3)' or 'sin' or 'gaussian'")

    def calc_posterior(self) -> None:
        S_inv = np.linalg.inv(self.S)

        y = self.y_train
        phi = self.phi
        if len(self.phi.shape) == 1:
            phi = self.phi.reshape(1, -1)
            y = y.reshape(1, 1)
        self.S = np.linalg.inv(S_inv + self.beta * phi.T @ phi)
        self.mu = self.S @ (S_inv @ self.mu + np.squeeze(self.beta * phi.T @ y))

    def sampling_params(self, n: int = 1, random_state: int = 0) -> np.ndarray:
        np.random.seed(random_state)
        return np.random.multivariate_normal(self.mu, self.S, n)

    def probability(self, x: pd.DataFrame):
        dist = multivariate_normal(mean=self.mu, cov=self.S)
        return dist.logpdf(x)

    def predict(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phi = self.x_to_phi(x_test, self.phi_type)

        if len(phi.shape) == 1:
            phi = phi.reshape(1, -1)
        pred = np.array([self.mu.T @ _phi for _phi in phi])  # 予測値(map推定なのでmuをそのまま使う)
        S_pred = np.array([(1 / self.beta) + _phi.T @ self.S @ _phi for _phi in phi])  # 予測値の分散

        # Above is a simple implementation.
        # This may be better if you want speed.
        # pred = self.mu @ phi.T
        # S_pred = (1 / self.beta) + np.diag(phi @ self.S @ phi.T)

        # RSME
        print(f"Predicted value: \n{pred}")
        print(f"Predicted variance: \n{S_pred}")
        print(f"RSME: {np.sqrt(np.mean((pred - y_test)**2))}")

        return pred, S_pred

    def predict_and_visualization(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        pred, S_pred = self.predict(x_test, y_test)

        plt.plot(x_test, pred, color="blue", label="predict")  # 予測分布の期待値
        plt.fill_between(
            x=x_test,
            y1=pred - np.sqrt(S_pred),
            y2=pred + np.sqrt(S_pred),
            color="#1e90ff",
            alpha=0.2,
            linestyle="dotted",
            label="$\u03bc_{pred} \\pm \u03c3_{pred}$",
        )  # 予測分布の標準偏差
        plt.scatter(self.x_train, self.y_train, color="#ff1493")  # 観測データ
        plt.suptitle("Predicted distribution", fontsize=15)
        plt.title(
            f"$N={self.x_train.shape[0]}, \u03a6: {self.phi_type}, M={self.M}, \u03c3={self.sigma}, \u03b1={self.alpha}$",
            loc="left",
            fontsize=11,
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim((min(y_test) - 3 * self.sigma, max(y_test) + 3 * self.sigma))  # y軸の表示範囲
        plt.legend()  # 凡例
        plt.grid()  # グリッド線
        plt.show()
