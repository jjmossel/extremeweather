import autograd.numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from autograd import hessian, grad


def gdp_logp(x, xi, sigma, mu=0.0):
    z = (x - mu) / sigma
    return -np.log(sigma) - (1.0 + 1.0 / xi) * np.log(1.0 + xi * z)


def gdp_returnlevel(returnperiod, mu, xi, sigma, p_mu, period=1.0):
    m = returnperiod * period
    return mu + (sigma / xi) * (np.power(m * p_mu, xi) - 1.0)


class GPDMLE:
    def __init__(self, u):
        self.u = u

        self.xi = None
        self.sigma = None

        self.x_exceed = None
        self.n_sample = None
        self.n_exceed = None

    def neg_ll(self, theta) -> float:
        xi, sigma = theta
        return -1.0 * np.sum(gdp_logp(self.x_exceed, xi, sigma))

    def fit(self, x_data):

        self.x_exceed = x_data[np.where(x_data > self.u)] - self.u

        self.n_sample = len(x_data)
        self.n_exceed = len(self.x_exceed)

        self.xi, self.sigma = sp.optimize.fmin(self.neg_ll, (0.001, 1.0))

    def p_u(self) -> float:
        return self.n_exceed / self.n_sample

    def p_u_var(self) -> float:
        p = self.p_u()
        return p * (1 - p) / self.n_sample

    def cov(self) -> np.ndarray:
        fisher_info = hessian(self.neg_ll)(np.array([self.xi, self.sigma]))
        cov = np.linalg.inv(fisher_info)
        return cov

    def get_params(self, include_ci=False) -> dict:

        params = {"xi": self.xi, "sigma": self.sigma}

        if include_ci:
            xi_se, sigma_se = np.sqrt(np.diag(self.cov()))

            ci = {
                "xi": (self.xi - 2 * xi_se, self.xi + 2 * xi_se),
                "sigma": (self.sigma - 2 * sigma_se, self.sigma + 2 * sigma_se),
            }

            return (params, ci)
        else:
            return params

    def return_level(self, return_period, period=1.0):

        return gdp_returnlevel(
            return_period, self.u, self.xi, self.sigma, self.p_u(), period=period
        )

    def _return_level_se(self, return_period, period=1.0):
        def rl(theta):
            xi, sigma, p_u = theta
            return gdp_returnlevel(return_period, self.u, xi, sigma, p_u, period=period)

        delta = grad(rl)(np.array([self.xi, self.sigma, self.p_u()]))

        cov = np.zeros((3, 3))
        cov[:2, :2] = self.cov()
        cov[2, 2] = self.p_u_var()

        return np.sqrt((delta.dot(cov)).dot(delta))

    def return_level_se(self, return_period, period=1.0):

        return np.vectorize(self._return_level_se)(return_period, period=period)

    def _return_periods(self, n=20, log_scale=True):

        n_start = self.n_sample / self.n_exceed
        n_end = self.n_sample

        if log_scale:
            rl = np.exp(np.linspace(np.log(n_start), np.log(n_end), n))
        else:
            rl = np.linspace(n_start, n_end, n)
        return rl

    def return_level_plot(self, include_ci=False, log_scale=True, period=1.0):

        rp = self._return_periods(n=50, log_scale=log_scale)
        rl = self.return_level(rp)

        rp_scaled = rp / period

        plt.plot(rp_scaled, rl)

        if include_ci:
            rl_se = self.return_level_se(rp)
            plt.fill_between(rp_scaled, rl - 2.0 * rl_se, rl + 2.0 * rl_se, alpha=0.25)

        plt.xlabel("return period")
        plt.ylabel("return level")

        if log_scale:
            plt.xscale("log")
        plt.show()
