from libc cimport math as cmath
from libc.stdint cimport *
from libc.stdlib cimport *
from cython.parallel cimport prange

def mktout_if(list mean_mu_alpha, double[:, ::1] errors, double par_gamma):
    cdef:
        size_t i, n
        double[4] exp
        double exp_par_gamma
        double mu10, mu11, mu20, mu21
        double alpha1, alpha2
        bint j_is_larger, j_is_smaller
        double threshold2, threshold3
        bint case1, case2, case3, case4, case5, case6
        double t0, t1, t2
        double t1_sum, t2_sum, p1_sum, p2_sum
        double c
    
    #compute the exp outside of the loop
    n = errors.shape[0]
    exp[0] = cmath.exp(<double>mean_mu_alpha[0])
    exp[1] = cmath.exp(<double>mean_mu_alpha[1])
    exp[2] = cmath.exp(<double>mean_mu_alpha[2])
    exp[3] = cmath.exp(<double>mean_mu_alpha[3])
    exp_par_gamma = cmath.exp(par_gamma)
    c = 168.0
    
    t1_sum = 0.0
    t2_sum = 0.0
    p1_sum = 0.0
    p2_sum = 0.0
    
    for i in range(n):
        mu10 = errors[i, 0] * exp[0]
        mu11 = exp_par_gamma * mu10
        mu20 = errors[i, 1] * exp[1]
        mu21 = exp_par_gamma * mu20
        alpha1 = errors[i, 2] * exp[2]
        alpha2 = errors[i, 3] * exp[3]
        
        j_is_larger = mu10 > mu20
        j_is_smaller = not j_is_larger
        threshold2 = (1 + mu10 * alpha1) / (c + alpha1)
        threshold3 = (1 + mu20 * alpha2) / (c + alpha2)
        
        if j_is_larger != 0:
            """
            case1 = mu10 < 1 / c
            case2 = mu21 >= threshold2
            case3 = not (case1 | case2)
            
            t0 = case1*c + case2 * (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) + case3 / threshold2
            t1 = case2 * (t0 * alpha1 * mu11 - alpha1) + case3 * (t0 * alpha1 * mu10 - alpha1)
            t2 = c - t0 - t1
            
            t1_sum += t1
            t2_sum += t2
            p1_sum += case2 + case3
            p2_sum += case2
            """
            t1_sum += mu10
            t2_sum += mu11
            p1_sum += alpha1
            p2_sum += alpha2

        else:
            """
            case4 = mu20 < 1 / c
            case5 = mu11 >= threshold3
            case6 = not (case4 | case5)
            
            t0 = case4 * c + case5 * (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) + case6 / threshold3
            t1 = case5 * (t0 * alpha1 * mu11 - alpha1)
            t2 = c - t0 - t1
            
            t1_sum += t1
            t2_sum += t2
            p1_sum += case5
            p2_sum += case5 + case6
            
            t1_sum += 4.0
            t2_sum += 3.0
            p1_sum += 2.0
            p2_sum += 1.0
            """
            t1_sum += mu11
            t2_sum += mu10
            p1_sum += alpha2
            p2_sum += alpha1
    
    return t1_sum/n, t2_sum/n, p1_sum/n, p2_sum/n

def mktout(list mean_mu_alpha, double[:, ::1] errors, double par_gamma):
    cdef:
        size_t i, n
        double[4] exp
        double exp_par_gamma
        double mu10, mu11, mu20, mu21
        double alpha1, alpha2
        bint j_is_larger, j_is_smaller
        double threshold2, threshold3
        bint case1, case2, case3, case4, case5, case6
        double t0, t1, t2
        double p12, p1, p2
        double t1_sum, t2_sum, p1_sum, p2_sum
        double c
    
    #compute the exp outside of the loop
    n = errors.shape[0]
    exp[0] = cmath.exp(<double>mean_mu_alpha[0])
    exp[1] = cmath.exp(<double>mean_mu_alpha[1])
    exp[2] = cmath.exp(<double>mean_mu_alpha[2])
    exp[3] = cmath.exp(<double>mean_mu_alpha[3])
    exp_par_gamma = cmath.exp(par_gamma)
    c = 168.0
    
    t1_sum = 0.0
    t2_sum = 0.0
    p1_sum = 0.0
    p2_sum = 0.0
    
    for i in range(n):
        mu10 = errors[i, 0] * exp[0]
        mu11 = exp_par_gamma * mu10
        mu20 = errors[i, 1] * exp[1]
        mu21 = exp_par_gamma * mu20
        alpha1 = errors[i, 2] * exp[2]
        alpha2 = errors[i, 3] * exp[3]
        
        j_is_larger = mu10 > mu20
        j_is_smaller = not j_is_larger
        threshold2 = (1 + mu10 * alpha1) / (c + alpha1)
        threshold3 = (1 + mu20 * alpha2) / (c + alpha2)
        
        """
        case1 = j_is_larger * (mu10 < 1 / c)
        case2 = j_is_larger * (mu21 >= threshold2)
        case3 = j_is_larger and not (case1 or case2)
        case4 = j_is_smaller * (mu20 < 1 / c)
        case5 = j_is_smaller * (mu11 >= threshold3)
        case6 = j_is_smaller and not (case4 or case5)
        
        t0 = case1*c+case2 * (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) +case3 / threshold2 +case4 * c +case5 * (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) + case6 / threshold3
        t1 = case2 * (t0 * alpha1 * mu11 - alpha1) +case3 * (t0 * alpha1 * mu10 - alpha1) +case5 * (t0 * alpha1 * mu11 - alpha1)
        t2 = c - t0 - t1
        
        p12 = case2 + case5
        p1 = case3 + p12
        p2 = case6 + p12
        
        t1_sum += t1
        t2_sum += t2
        p1_sum += p1
        p2_sum += p2
        """
        t1_sum += mu10
        t2_sum += mu11
        p1_sum += alpha1
        p2_sum += alpha2
    
    return t1_sum/n, t2_sum/n, p1_sum/n, p2_sum/n

ctypedef struct Vec4:
    double a
    double b
    double c
    double d

def outer_loop(list mean_mu_alpha, double[:, ::1] errors, double par_gamma, size_t n):
    cdef:
        size_t i
        Vec4 mean_vec
        Vec4 out
    
    mean_vec.a = <double>(mean_mu_alpha[0])
    mean_vec.b = <double>(mean_mu_alpha[1])
    mean_vec.c = <double>(mean_mu_alpha[2])
    mean_vec.d = <double>(mean_mu_alpha[3])
    
    with nogil:
        for i in prange(n):
            cy_mktout(&out, &mean_vec, errors, par_gamma)
    return out

cdef void cy_mktout(Vec4 *out, Vec4 *mean_mu_alpha, double[:, ::1] errors, double par_gamma) nogil:
    cdef:
        size_t i, n
        double[4] exp
        double exp_par_gamma
        double mu10, mu11, mu20, mu21
        double alpha1, alpha2
        bint j_is_larger, j_is_smaller
        double threshold2, threshold3
        bint case1, case2, case3, case4, case5, case6
        double t0, t1, t2
        double p12, p1, p2
        double t1_sum, t2_sum, p1_sum, p2_sum
        double c
    
    #compute the exp outside of the loop
    n = errors.shape[0]
    exp[0] = cmath.exp(mean_mu_alpha.a)
    exp[1] = cmath.exp(mean_mu_alpha.b)
    exp[2] = cmath.exp(mean_mu_alpha.c)
    exp[3] = cmath.exp(mean_mu_alpha.d)
    exp_par_gamma = cmath.exp(par_gamma)
    c = 168.0
    
    t1_sum = 0.0
    t2_sum = 0.0
    p1_sum = 0.0
    p2_sum = 0.0
    
    for i in range(n):
        mu10 = errors[i, 0] * exp[0]
        mu11 = exp_par_gamma * mu10
        mu20 = errors[i, 1] * exp[1]
        mu21 = exp_par_gamma * mu20
        alpha1 = errors[i, 2] * exp[2]
        alpha2 = errors[i, 3] * exp[3]
        
        j_is_larger = mu10 > mu20
        j_is_smaller = not j_is_larger
        threshold2 = (1 + mu10 * alpha1) / (c + alpha1)
        threshold3 = (1 + mu20 * alpha2) / (c + alpha2)
        
        case1 = j_is_larger * (mu10 < 1 / c)
        case2 = j_is_larger * (mu21 >= threshold2)
        case3 = j_is_larger and not (case1 | case2)
        case4 = j_is_smaller * (mu20 < 1 / c)
        case5 = j_is_smaller * (mu11 >= threshold3)
        case6 = j_is_smaller and not (case4 | case5)
        
        t0 = case1*c+case2 * (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) +case3 / threshold2 +case4 * c +case5 * (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) + case6 / threshold3
        t1 = case2 * (t0 * alpha1 * mu11 - alpha1) +case3 * (t0 * alpha1 * mu10 - alpha1) +case5 * (t0 * alpha1 * mu11 - alpha1)
        t2 = c - t0 - t1
        
        p12 = case2 + case5
        p1 = case3 + p12
        p2 = case6 + p12
        
        t1_sum += t1
        t2_sum += t2
        p1_sum += p1
        p2_sum += p2
    
    out.a = t1_sum/n
    out.b = t2_sum/n
    out.c = p1_sum/n
    out.d = p2_sum/n

def outer_loop_if(list mean_mu_alpha, double[:, ::1] errors, double par_gamma, size_t n):
    cdef:
        size_t i
        Vec4 mean_vec
        Vec4 out
    
    mean_vec.a = <double>(mean_mu_alpha[0])
    mean_vec.b = <double>(mean_mu_alpha[1])
    mean_vec.c = <double>(mean_mu_alpha[2])
    mean_vec.d = <double>(mean_mu_alpha[3])
    
    with nogil:
        for i in prange(n):
            cy_mktout_if(&out, &mean_vec, errors, par_gamma)
    return out

cdef void cy_mktout_if(Vec4 *out, Vec4 *mean_mu_alpha, double[:, ::1] errors, double par_gamma) nogil:
    cdef:
        size_t i, n
        double[4] exp
        double exp_par_gamma
        double mu10, mu11, mu20, mu21
        double alpha1, alpha2
        bint j_is_larger
        double threshold2, threshold3
        bint case1, case2, case3, case4, case5, case6
        double t0, t1, t2
        double p12, p1, p2
        double t1_sum, t2_sum, p1_sum, p2_sum
        double c
    
    #compute the exp outside of the loop
    n = errors.shape[0]
    exp[0] = cmath.exp(mean_mu_alpha.a)
    exp[1] = cmath.exp(mean_mu_alpha.b)
    exp[2] = cmath.exp(mean_mu_alpha.c)
    exp[3] = cmath.exp(mean_mu_alpha.d)
    exp_par_gamma = cmath.exp(par_gamma)
    c = 168.0
    
    t1_sum = 0.0
    t2_sum = 0.0
    p1_sum = 0.0
    p2_sum = 0.0
    
    for i in range(n):
        mu10 = errors[i, 0] * exp[0]
        mu11 = exp_par_gamma * mu10
        mu20 = errors[i, 1] * exp[1]
        mu21 = exp_par_gamma * mu20
        alpha1 = errors[i, 2] * exp[2]
        alpha2 = errors[i, 3] * exp[3]
        
        j_is_larger = mu10 > mu20
        j_is_smaller = not j_is_larger
        threshold2 = (1 + mu10 * alpha1) / (c + alpha1)
        threshold3 = (1 + mu20 * alpha2) / (c + alpha2)
        
        if j_is_larger:
            case1 = mu10 < 1 / c
            case2 = mu21 >= threshold2
            case3 = not (case1 | case2)
            
            t0 = case1*c + case2 * (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) + case3 / threshold2
            t1 = case2 * (t0 * alpha1 * mu11 - alpha1) + case3 * (t0 * alpha1 * mu10 - alpha1)
            t2 = c - t0 - t1
            
            t1_sum += t1
            t2_sum += t2
            p1_sum += case2 + case3
            p2_sum += case2
            
        else:
            case4 = mu20 < 1 / c
            case5 = mu11 >= threshold3
            case6 = not (case4 | case5)
            
            t0 = case4 * c + case5 * (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2) + case6 / threshold3
            t1 = case5 * (t0 * alpha1 * mu11 - alpha1)
            t2 = c - t0 - t1
            
            t1_sum += t1
            t2_sum += t2
            p1_sum += case5
            p2_sum += case5 + case6
    
    out.a = t1_sum/n
    out.b = t2_sum/n
    out.c = p1_sum/n
    out.d = p2_sum/n

