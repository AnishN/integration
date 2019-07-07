import integrate
import numpy as np

par_mu_rho = 0.8
par_alpha_rho = 0.7
cov_epsilon = [[1, par_mu_rho], [par_mu_rho, 1]]
cov_nu = [[1, par_alpha_rho], [par_alpha_rho, 1]]
nrows = 10000 
np.random.seed(123)
epsilon_sim = np.random.multivariate_normal([0, 0], cov_epsilon, nrows)
nu_sim = np.random.multivariate_normal([0, 0], cov_nu, nrows)
errors = np.concatenate((epsilon_sim, nu_sim), axis=1)
errors = np.exp(errors)

print(errors)
print(errors.shape)
out = integrate.mktout([-6,-6,-1,-1], errors, -0.7)
print(out)
