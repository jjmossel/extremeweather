import autograd.numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from autograd import hessian, grad


def gpd_logp(x, xi, sigma, mu=0.0):
    z = (x - mu) / sigma
    return -np.log(sigma) - (1.0 + 1.0 / xi) * np.log(1.0 + xi * z)


def gpd_pdf(x, xi, sigma, mu=0.0):
    return np.exp(gpd_logp(x, xi, sigma, mu=mu))


def gpd_returnlevel(returnperiod, mu, xi, sigma, p_mu, period=1.0):
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

    def _neg_ll(self, theta) -> float:
        xi, sigma = theta
        return -1.0 * np.sum(gpd_logp(self.x_exceed, xi, sigma))

    def fit(self, x_data, disp=False):

        self.x_exceed = x_data[np.where(x_data > self.u)] - self.u

        self.n_sample = len(x_data)
        self.n_exceed = len(self.x_exceed)

        self.xi, self.sigma = sp.optimize.fmin(self._neg_ll, (0.001, 1.0), disp=disp)

    def p_u(self) -> float:
        return self.n_exceed / self.n_sample

    def p_u_var(self) -> float:
        p = self.p_u()
        return p * (1 - p) / self.n_sample

    def cov(self) -> np.ndarray:
        fisher_info = hessian(self._neg_ll)(np.array([self.xi, self.sigma]))
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

        return gpd_returnlevel(
            return_period, self.u, self.xi, self.sigma, self.p_u(), period=period
        )

    def _return_level_se(self, return_period, period=1.0):
        def rl(theta):
            xi, sigma, p_u = theta
            return gpd_returnlevel(return_period, self.u, xi, sigma, p_u, period=period)

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

    def pdf(self, x):
        return gpd_pdf(x, self.xi, self.sigma, mu=0.0)

    def dist_plot(self, n=20):

        x0 = 0.0
        x1 = np.quantile(self.x_exceed, 0.999)
        x_vals = np.linspace(x0, x1, n)

        pdf = self.pdf(x_vals)

        plt.hist(self.x_exceed, bins=n, range=(x0, x1), density=True)
        plt.plot(x_vals, pdf)


def gev_logp(x, mu, sigma, xi):

    z = (x - mu) / sigma
    t = 1.0 + xi * z

    return -np.power(t, -1.0 / xi) - (1.0 / xi + 1.0) * np.log(t) - np.log(sigma)


def gev_pdf(x, mu, sigma, xi):
    return np.exp(gev_logp(x, mu, sigma, xi))


def gev_returnlevel(rl, mu, sigma, xi):
    p = 1.0 / rl
    return mu - (sigma / xi) * (1 - np.power(-np.log(1.0 - p), -xi))


def gev_sf(x, mu, sigma, xi):
    z = (x - mu) / sigma
    return 1.0 - np.exp(-np.power(1 + xi * z, -1.0 / xi))


class GEVMLE:
    def __init__(self, period=100):

        self.period = period
        self.x_max = None

        self.mu = None
        self.sigma = None
        self.xi = None

    def _fit_x_max(self, x):
        n_blocks = len(x) // self.period
        self.x_max = np.max(np.array_split(x, n_blocks), axis=1)

    def _neg_ll(self, theta) -> float:
        mu, sigma, xi = theta
        return -1.0 * np.sum(gev_logp(self.x_max, mu, sigma, xi))

    def fit(self, x_data, disp=False):

        self._fit_x_max(x_data)

        neg_xi, mu, sigma = sp.stats.genextreme.fit(self.x_max)

        self.mu = mu
        self.sigma = sigma
        self.xi = -neg_xi

    def cov(self) -> np.ndarray:
        fisher_info = hessian(self._neg_ll)(np.array([self.mu, self.sigma, self.xi]))
        cov = np.linalg.inv(fisher_info)
        return cov

    def get_params(self, include_ci=False) -> dict:

        params = {"mu": self.mu, "sigma": self.sigma, "xi": self.xi}

        if include_ci:
            mu_se, sigma_se, xi_se = np.sqrt(np.diag(self.cov()))

            ci = {
                "mu": (self.mu - 2 * mu_se, self.mu + 2 * mu_se),
                "sigma": (self.sigma - 2 * sigma_se, self.sigma + 2 * sigma_se),
                "xi": (self.xi - 2 * xi_se, self.xi + 2 * xi_se),
            }

            return (params, ci)
        else:
            return params

    def return_level(self, return_period):

        return gev_returnlevel(
            return_period,
            self.mu,
            self.sigma,
            self.xi,
        )

    def _return_level_se(self, return_period):
        def rl(theta):
            mu, sigma, xi = theta
            return gev_returnlevel(return_period, mu, sigma, xi)

        delta = grad(rl)(np.array([self.mu, self.sigma, self.xi]))

        cov = self.cov()

        return np.sqrt((delta.dot(cov)).dot(delta))

    def return_level_se(self, return_period):

        return np.vectorize(self._return_level_se)(return_period)

    def _return_periods(self, n=20, log_scale=True):

        n_start = 2
        n_end = len(self.x_max)

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

    def pdf(self, x):
        return gev_pdf(x, self.mu, self.sigma, self.xi)

    def dist_plot(self, n=20):

        x0 = np.min(self.x_max)
        x1 = np.quantile(self.x_max, 0.999)
        x_vals = np.linspace(x0, x1, n)

        pdf = self.pdf(x_vals)

        plt.hist(self.x_max, bins=n, range=(x0, x1), density=True)
        plt.plot(x_vals, pdf)

    def sf(self, x):
        return gev_sf(x, self.mu, self.sigma, self.xi)

    def return_period(self, return_level):
        return 1.0 / self.sf(return_level)

    def _return_period_se(self, return_level):
        def rp(theta):
            mu, sigma, xi = theta
            return 1.0 / gev_sf(return_level, mu, sigma, xi)

        delta = grad(rp)(np.array([self.mu, self.sigma, self.xi]))

        cov = self.cov()

        return np.sqrt((delta.dot(cov)).dot(delta))

    def return_period_se(self, return_level):

        return np.vectorize(self._return_period_se)(return_level)


class GEVMLE_ts(GEVMLE):
    def __init__(self):
        # only anual for now

        super().__init__(365.25)

    def _fit_x_max(self, ts):

        ts = ts.copy()
        ts = ts.to_frame("value")
        ts["year"] = ts.index.year

        def period_max(ts):
            if ts.count() >= 330:
                return ts.max()
            else:
                return np.nan

        ts_max = ts.groupby("year")["value"].apply(period_max)

        self.x_max = ts_max.dropna().values
