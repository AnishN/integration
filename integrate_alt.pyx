from libc cimport math as cmath
from libc.stdint cimport *
from libc.stdlib cimport *
from cython.parallel cimport prange

cdef double c = 168.0
cdef double threshold1 = -cmath.log(c)

def mktout_alt(list mean_mu_alpha, double[:, ::1] errors, double par_gamma):
    cdef:
        size_t i, n
        double [4] err#error
        double [4] mma#mean_mu_alpha values
        double t0_sum, t1_sum, t2_sum
        double mu10, mu11, mu20, mu21
        double threshold2, threshold3
    
    mma[0] = <double>mean_mu_alpha[0]
    mma[1] = <double>mean_mu_alpha[1]
    mma[2] = <double>mean_mu_alpha[2]
    mma[3] = <double>mean_mu_alpha[3]
    
    t0_sum = 0.0
    t1_sum = 0.0
    t2_sum = 0.0
    
    n = errors.shape[0]
    for i in range(n):
        err[0] = errors[i, 0]
        err[1] = errors[i, 1]
        err[2] = errors[i, 2]
        err[3] = errors[i, 3]
        
        if err[0] + mma[0] >= err[1] + mma[1]:
            if err[0] + mma[0] < threshold1:
                t0_sum += c
                #t1_sum += 0
                #t2_sum += 0
            else:
                mu21 = cmath.exp(err[1] + par_gamma + mma[1])
                mu10 = cmath.exp(err[0] + mma[0])
                alpha1 = cmath.exp(err[2] + mma[2])
                alpha2 = cmath.exp(err[3] + mma[3])
                threshold2 = (1 + mu10 * alpha1) / (c + alpha1)
                if mu21 >= threshold2:
                    mu11 = cmath.exp(err[0] + par_gamma + mma[0])
                    t0 = (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2)
                    t0_sum += t0
                    t1_sum += (t0 * mu11 - 1) * alpha1
                    t2_sum += (t0 * mu21 - 1) * alpha2
                else:
                    t0 = (c + alpha1) / (1 + mu10 * alpha1)
                    t0_sum += t0
                    t1_sum += (t0 * mu10 - 1) * alpha1
                    #t2_sum += 0
        else:
            if err[1] + mma[1] < threshold1:
                t0_sum += c
                #t1_sum += 0
                #t2_sum += 0
            else:
                mu11 = cmath.exp(err[0] + par_gamma + mma[0])
                mu20 = cmath.exp(err[1] + mma[1])
                alpha1 = cmath.exp(err[2] + mma[2])
                alpha2 = cmath.exp(err[3] + mma[3])
                threshold3 = (1 + mu20 * alpha2) / (c + alpha2)
                
                if mu11 >= threshold3:
                    mu21 = cmath.exp(err[0] + par_gamma + mma[0])
                    t0 = (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2)
                    t0_sum += t0
                    t1_sum += (t0 * mu11 - 1) * alpha1
                    t2_sum += (t0 * mu21 - 1) * alpha2
                else:
                    t0 = (c + alpha2) / (1 + mu20 * alpha2)
                    t0_sum += t0
                    #t1_sum += 0
                    t2_sum += (t0 * mu20 - 1) * alpha2
                    
    return t0_sum/n, t1_sum/n, t2_sum/n

ctypedef struct Vec3:
    double a
    double b
    double c
    
ctypedef struct Vec4:
    double a
    double b
    double c
    double d

def outer_loop_alt(list mean_mu_alpha, double[:, ::1] errors, double par_gamma, size_t n):
    cdef:
        size_t i
        Vec4 mma
        Vec3 out
    
    mma.a = <double>mean_mu_alpha[0]
    mma.b = <double>mean_mu_alpha[1]
    mma.c = <double>mean_mu_alpha[2]
    mma.d = <double>mean_mu_alpha[3]
    
    with nogil:
        for i in prange(n):
            cy_mktout_alt(&out, &mma, errors, par_gamma)
    return out

cdef void cy_mktout_alt(Vec3 *out, Vec4 *mma, double[:, ::1] errors, double par_gamma) nogil:
    cdef:
        size_t i, n
        double [4] err#error
        double t0_sum, t1_sum, t2_sum
        double mu10, mu11, mu20, mu21
        double threshold2, threshold3
    
    t0_sum = 0.0
    t1_sum = 0.0
    t2_sum = 0.0
    
    n = errors.shape[0]
    for i in range(n):
        err[0] = errors[i, 0]
        err[1] = errors[i, 1]
        err[2] = errors[i, 2]
        err[3] = errors[i, 3]
        
        if err[0] + mma.a >= err[1] + mma.b:
            if err[0] + mma.a < threshold1:
                t0_sum += c
                #t1_sum += 0
                #t2_sum += 0
            else:
                mu21 = cmath.exp(err[1] + par_gamma + mma.b)
                mu10 = cmath.exp(err[0] + mma.a)
                alpha1 = cmath.exp(err[2] + mma.c)
                alpha2 = cmath.exp(err[3] + mma.d)
                threshold2 = (1 + mu10 * alpha1) / (c + alpha1)
                if mu21 >= threshold2:
                    mu11 = cmath.exp(err[0] + par_gamma + mma.a)
                    t0 = (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2)
                    t0_sum += t0
                    t1_sum += (t0 * mu11 - 1) * alpha1
                    t2_sum += (t0 * mu21 - 1) * alpha2
                else:
                    t0 = (c + alpha1) / (1 + mu10 * alpha1)
                    t0_sum += t0
                    t1_sum += (t0 * mu10 - 1) * alpha1
                    #t2_sum += 0
        else:
            if err[1] + mma.b < threshold1:
                t0_sum += c
                #t1_sum += 0
                #t2_sum += 0
            else:
                mu11 = cmath.exp(err[0] + par_gamma + mma.a)
                mu20 = cmath.exp(err[1] + mma.b)
                alpha1 = cmath.exp(err[2] + mma.c)
                alpha2 = cmath.exp(err[3] + mma.d)
                threshold3 = (1 + mu20 * alpha2) / (c + alpha2)
                
                if mu11 >= threshold3:
                    mu21 = cmath.exp(err[0] + par_gamma + mma.a)
                    t0 = (c + alpha1 + alpha2) / (1 + mu11 * alpha1 + mu21 * alpha2)
                    t0_sum += t0
                    t1_sum += (t0 * mu11 - 1) * alpha1
                    t2_sum += (t0 * mu21 - 1) * alpha2
                else:
                    t0 = (c + alpha2) / (1 + mu20 * alpha2)
                    t0_sum += t0
                    #t1_sum += 0
                    t2_sum += (t0 * mu20 - 1) * alpha2
                    
    out.a = t0_sum/n
    out.b = t1_sum/n
    out.c = t2_sum/n
