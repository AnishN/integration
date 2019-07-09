from libc cimport math as cmath
from libc.stdint cimport *
from libc.stdlib cimport *
from cython.parallel cimport prange

def mktout_full(double[:] mean_mu_alpha, double[:, ::1] errors, double par_gamma):
    cdef:
        size_t i, n
        double[4] exp
        double exp_par_gamma
        double mu10, mu11, mu20, mu21
        double alpha1, alpha2
        double threshold2, threshold3
        double t0, t1, t2
        double t1_sum, t2_sum, p1_sum, p2_sum, p12_sum
        double c

    #compute the exp outside of the loop
    n = errors.shape[0]
    exp[0] = cmath.exp(mean_mu_alpha[0])
    exp[1] = cmath.exp(mean_mu_alpha[1])
    exp[2] = cmath.exp(mean_mu_alpha[2])
    exp[3] = cmath.exp(mean_mu_alpha[3])
    exp_par_gamma = cmath.exp(par_gamma)
    c = 168.0

    t1_sum = 0.0
    t2_sum = 0.0
    p1_sum = 0.0
    p2_sum = 0.0
    p12_sum = 0.0

    for i in range(n) :
        mu10 = errors[i, 0] * exp[0]
        mu20 = errors[i, 1] * exp[1]
        if (mu10 >= mu20):
            if (mu10 >= 1/c) :
                mu21 = exp_par_gamma * mu20
                alpha1 = errors[i, 2] * exp[2]
                alpha2 = errors[i, 3] * exp[3]
                threshold2 = (1 + mu10 * alpha1) / (c + alpha1)
                if (mu21 >= threshold2):
                    mu11 = exp_par_gamma * mu10
                    t0 =  (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2)
                    t1 = (t0 * alpha1 * mu11 - alpha1)
                    t1_sum += t1
                    t2_sum += c - t0 - t1
                    p1_sum += 1
                    p2_sum += 1
                    p12_sum += 1
                else :
                    t1_sum += ((1/threshold2) * alpha1 * mu10 - alpha1)
                    p1_sum += 1
        else :
            if (mu20 >= 1/c) :
                mu11 = exp_par_gamma * mu10
                alpha1 = errors[i, 2] * exp[2]
                alpha2 = errors[i, 3] * exp[3]
                threshold3 = (1 + mu20 * alpha2) / (c + alpha2)
                if (mu11 >= threshold3):
                    mu21 = exp_par_gamma * mu20
                    t0 =  (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2)
                    t1 = (t0 * alpha1 * mu11 - alpha1)
                    t1_sum += t1
                    t2_sum += c - t0 - t1
                    p1_sum += 1
                    p2_sum += 1
                    p12_sum += 1
                else :
                    t2_sum += ((1/threshold3) * alpha2 * mu20 - alpha2)
                    p2_sum += 1

    return t1_sum/n, t2_sum/n, p1_sum/n, p2_sum/n, p12_sum/n

def outer_loop_full(double[:] out, double[:] mean_mu_alpha, double[:, ::1] errors, double par_gamma, size_t n):
    cdef size_t i
    
    with nogil:
        for i in prange(n):
            cy_mktout_full(out, mean_mu_alpha, errors, par_gamma)
            
cdef void cy_mktout_full(double[:] out, double[:] mean_mu_alpha, double[:, ::1] errors, double par_gamma) nogil:
    cdef:
        size_t i, n
        double[4] exp
        double exp_par_gamma
        double mu10, mu11, mu20, mu21
        double alpha1, alpha2
        double threshold2, threshold3
        double t0, t1, t2
        double t1_sum, t2_sum, p1_sum, p2_sum, p12_sum
        double c

    #compute the exp outside of the loop
    n = errors.shape[0]
    exp[0] = cmath.exp(mean_mu_alpha[0])
    exp[1] = cmath.exp(mean_mu_alpha[1])
    exp[2] = cmath.exp(mean_mu_alpha[2])
    exp[3] = cmath.exp(mean_mu_alpha[3])
    exp_par_gamma = cmath.exp(par_gamma)
    c = 168.0

    t1_sum = 0.0
    t2_sum = 0.0
    p1_sum = 0.0
    p2_sum = 0.0
    p12_sum = 0.0

    for i in range(n) :
        mu10 = errors[i, 0] * exp[0]
        mu20 = errors[i, 1] * exp[1]
        if (mu10 >= mu20):
            if (mu10 >= 1/c) :
                mu21 = exp_par_gamma * mu20
                alpha1 = errors[i, 2] * exp[2]
                alpha2 = errors[i, 3] * exp[3]
                threshold2 = (1 + mu10 * alpha1) / (c + alpha1)
                if (mu21 >= threshold2):
                    mu11 = exp_par_gamma * mu10
                    t0 =  (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2)
                    t1 = (t0 * alpha1 * mu11 - alpha1)
                    t1_sum += t1
                    t2_sum += c - t0 - t1
                    p1_sum += 1
                    p2_sum += 1
                    p12_sum += 1
                else :
                    t1_sum += ((1/threshold2) * alpha1 * mu10 - alpha1)
                    p1_sum += 1
        else :
            if (mu20 >= 1/c) :
                mu11 = exp_par_gamma * mu10
                alpha1 = errors[i, 2] * exp[2]
                alpha2 = errors[i, 3] * exp[3]
                threshold3 = (1 + mu20 * alpha2) / (c + alpha2)
                if (mu11 >= threshold3):
                    mu21 = exp_par_gamma * mu20
                    t0 =  (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2)
                    t1 = (t0 * alpha1 * mu11 - alpha1)
                    t1_sum += t1
                    t2_sum += c - t0 - t1
                    p1_sum += 1
                    p2_sum += 1
                    p12_sum += 1
                else :
                    t2_sum += ((1/threshold3) * alpha2 * mu20 - alpha2)
                    p2_sum += 1

    out[0] = t1_sum/n
    out[1] = t2_sum/n
    out[2] = p1_sum/n
    out[3] = p2_sum/n
    out[4] = p12_sum/n
