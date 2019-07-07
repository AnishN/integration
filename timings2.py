import time

import integrate
import numpy as np
import numexpr as ne
import math
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

def mktout(mean_mu_alpha, errors, par_gamma):
    mu10 = errors[:, 0] * math.exp(mean_mu_alpha[0])
    mu11 = math.exp(par_gamma) * mu10  # mu with gamma
    mu20 = errors[:, 1] * math.exp(mean_mu_alpha[1])
    mu21 = math.exp(par_gamma) * mu20
    alpha1 = errors[:, 2] * math.exp(mean_mu_alpha[2])
    alpha2 = errors[:, 3] * math.exp(mean_mu_alpha[3])

    j_is_larger = (mu10 > mu20)
    threshold2 = (1 + mu10 * alpha1) / (168 + alpha1)
    j_is_smaller = ~j_is_larger
    threshold3 = (1 + mu20 * alpha2) / (168 + alpha2)
    case1 = j_is_larger * (mu10 < 1 / 168)
    case2 = j_is_larger * (mu21 >= threshold2)
    case3 = j_is_larger ^ (case1 | case2)
    case4 = j_is_smaller * (mu20 < 1 / 168)
    case5 = j_is_smaller * (mu11 >= threshold3)
    case6 = j_is_smaller ^ (case4 | case5)
    t0 = ne.evaluate("case1*168+case2 * (168 + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) +case3 / threshold2 +case4 * 168 +case5 * (168 + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) + case6 / threshold3")
    t1 = ne.evaluate("case2 * (t0 * alpha1 * mu11 - alpha1) +case3 * (t0 * alpha1 * mu10 - alpha1) +case5 * (t0 * alpha1 * mu11 - alpha1)")
    t2 = 168 - t0 - t1
    p12 = case2 + case5
    p1 = case3 + p12
    p2 = case6 + p12
    return t1.sum()/10000, t2.sum()/10000, p1.sum()/10000, p2.sum()/10000
    
n = 1000

start = time.time()
out = integrate.outer_loop([-6,-6,-1,-1], errors, -0.7, n)
end = time.time()
print(end - start)

start = time.time()
for i in range(n):
    out = integrate.mktout([-6,-6,-1,-1], errors, -0.7)
end = time.time()
print(end - start)

start = time.time()
for i in range(n):
    out = mktout([-6,-6,-1,-1], errors, -0.7)
end = time.time()
print(end - start)
