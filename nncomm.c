/*
 *   Copyright (C), 2017-1, Tech. Co., Ltd.
 *   File name   : nncomm.c
 *   Author      : vincent
 *   Version     : 0.9
 *   Date        : 2017.6
 *   Description : cnn
 *   GitHub      : https://github.com/optimizevin
 */

#include "nncomm.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <memory.h>

#define sigmoid_exp fastexp
#define Relu_def(x) x>0?x:0
//noisyRelu
//leakyRelu
//

inline float fastexp(float x)
{
    /* e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) + (x^4 / 4!) */
    float sum = 1 + x;
    float n = x;
    float d = 1;
    float i;

    for(i = 2; i < 100; i++) {
        n *= x;
        d *= i;
        sum += n / d;
    }

    return sum;
}


inline float generateGaussianNoise(const float mean, const float stdDev)
{
    static int hasSpare = 0;
    static float spare;

    if(hasSpare) {
        hasSpare = 0;
        return mean + stdDev * spare;
    }

    hasSpare = 1;
    static float u, v, s;
    do {
        u = (rand() / ((float) RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((float) RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while( (s >= 1.0) || (s == 0.0) );

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + stdDev * u * s;
}

inline float sigmoid(const float x)
{
    return 1.00 / (1 + sigmoid_exp(0 - x));
}

inline  float* randf(const uint32_t nsize, const float stddev)
{
    float *ret = (float*)calloc(nsize, sizeof(float));
    for(uint32_t i = 0; i < nsize; i++) {
        ret[i] = generateGaussianNoise(0.f, stddev);
    }

    return ret;
}

inline  void bias(float *pfloat, const  uint32_t nsize, const float stddev)
{
    for(uint32_t i = 0; i < nsize; i++) {
        pfloat[i] += stddev;
    }
}


inline float Relu(const float *pf, uint32_t len)
{
    float t = 0.f;
    for(uint32_t i = 0; i < len; i++) {
        float  m = Relu_def(pf[i]);
        t =  t > m ? t : m;
    }

    return  t;
}

inline void Dropout(const float *src, const uint32_t len, float keep_prob, float *out)
{
    for(int i = 0; i < len; i++) {
        float tmp = (rand() / ((float)RAND_MAX)) ;
        (tmp > (1 - keep_prob)) ? (out[i] = src[i]) : (out[i] = 0);
    }
}


inline float *MatrixAdd(const float *a, const float *b, uint32_t r, uint32_t c)
{
    float * pret = (float*)calloc(r * c, sizeof(float));
    for(uint32_t i = 0; i < r; i++) {
        for(uint32_t j = 0; j < c; j++) {
            ((float (*)[c])pret)[i][j] = ((float (*)[c])(a))[i][j] + ((float (*)[c])(b))[i][j];
        }
    }

    return pret;
}

inline void intMatrixMutiply(const uint32_t *a, const uint32_t *b, uint32_t *c, uint32_t arow, uint32_t acol, uint32_t bcol)
{
    for(uint32_t i = 0; i < arow; i++) {
        uint32_t *ptr_c = c + i * bcol;
        uint32_t *ptr_a = (uint32_t*)a + i * acol;
        for(uint32_t j = 0; j < bcol; j++) {
            for(uint32_t k = 0; k < acol; k++) {
                ptr_c[j] += ptr_a[k] * b[k * bcol + j];
            }
        }
    }
}

inline void floatMatrixMutiply(const float *a, const float *b, float *c, uint32_t arow, uint32_t acol, uint32_t bcol)
{
    for(uint32_t i = 0; i < arow; i++) {
        float *ptr_c = c + i * bcol;
        float *ptr_a = (float*)a + i * acol;
        for(uint32_t j = 0; j < bcol; j++) {
            for(uint32_t k = 0; k < acol; k++) {
                ptr_c[j] += ptr_a[k] * b[k * bcol + j];
            }
        }
    }
}


inline void max_pool(float *src, uint32_t rows, uint32_t cols, uint32_t pool_size, float*pout)
{
    float(*pSrc)[cols] = (float(*)[cols])src;
    float(*pOut)[cols - pool_size + 1] = (float(*)[cols - pool_size + 1])pout;

    for(uint32_t i = 0; i <= (rows - pool_size); i++) {
        for(uint32_t j = 0; j <= (cols - pool_size); j++) {
            float tmp = 0.f;
            for(uint32_t ip = 0; ip < pool_size; ip++) {
                for(uint32_t jp = 0; jp < pool_size; jp++) {
                    tmp =  MAX(pSrc[ip + i][jp + j], tmp);
                }
            }
            pOut[i][j] =  tmp;
        }
    }
}


inline float reduce_ment(const float *src, const uint32_t len)
{
    float ret = 0.f;
    for(uint32_t i = 0 ; i < len; i++) {
        ret += src[i];
    }
    ret /= len;
    return ret;
}


inline void foreach_log(float *src, const uint32_t len)
{
    for(uint32_t i = 0 ; i < len; i++) {
        src[i] = log(src[i]);
    }
}


inline void softMax(float *src, uint32_t rows, uint32_t cols)
{
    for (uint32_t i = 0; i < rows; ++i) {
        float max = 0.0;
        float sum = 0.0;
        for (uint32_t j = 0; j < cols; j++) {
            if (max < src[j + i * cols])
                max = src[j + i * cols];
        }
        for (uint32_t j = 0; j < cols; j++) {
            src[j + i * cols] = exp(src[j + i * cols] - max);
            sum += src[j + i * cols];
        }
        for (uint32_t j = 0; j < cols; j++) {
            src[j + i * cols] /= sum;
        }
    }
}

inline void softMax_cross_entropy_with_logits(const float *labels, const float *logits,
        uint32_t rows, uint32_t cols, float *pOut)
{
    float tmp[rows * cols];
    memcpy(tmp, logits, rows * cols * sizeof(float));
    softMax(tmp, rows, cols);
    const  uint32_t bsize = sizeof(tmp) / sizeof(tmp[0]);
    foreach_log(tmp, bsize);

    float bout[bsize] ;
    /*memset(bout, 0x0, bsize);*/

    for(uint32_t i = 0; i < bsize; i++) {
        bout[i] = tmp[i] * labels[i];
    }

    float(*pBout)[cols] = (float(*)[cols])bout;
    for(uint32_t i = 0; i < rows; i++) {
        float  t = 0.f;
        for(uint32_t j = 0; j < cols; j++) {
            t += pBout[i][j];
        }
        pOut[i] = -t;
    }

}


/*SGD with momentum*/
inline void SGD()
{
}

inline void SGD_Momentum()
{
}


/************************************************
   "Adam optimizer.

   Default parameters follow those provided in the original paper.

       lr: float >= 0. Learning rate.
       beta_1: float, 0 < beta < 1. Generally close to 1.
       beta_2: float, 0 < beta < 1. Generally close to 1.
       epsilon: float >= 0. Fuzz factor.
       decay: float >= 0. Learning rate decay over each update.

   # References
   - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
     learning_rate=0.001,
     beta1=0.9, beta2=0.999, epsilon=1e-08,
     use_locking=False,

************************************************/

inline  void AdamOptimizer(const float lr,const float beta_1,const float beta_2,
        const float epsilon,const float decay)
{
}

/*class Adam(Optimizer):*/

/*def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,*/
/*epsilon=1e-8, decay=0., **kwargs):*/
/*super(Adam, self).__init__(**kwargs)*/
/*with K.name_scope(self.__class__.__name__):*/
/*self.iterations = K.variable(0, dtype='int64', name='iterations')*/
/*self.lr = K.variable(lr, name='lr')*/
/*self.beta_1 = K.variable(beta_1, name='beta_1')*/
/*self.beta_2 = K.variable(beta_2, name='beta_2')*/
/*self.decay = K.variable(decay, name='decay')*/
/*self.epsilon = epsilon*/
/*self.initial_decay = decay*/

/*@interfaces.legacy_get_updates_support*/
/*def get_updates(self, loss, params):*/
/*grads = self.get_gradients(loss, params)*/
/*self.updates = [K.update_add(self.iterations, 1)]*/

/*lr = self.lr*/
/*if self.initial_decay > 0:*/
/*lr *= (1. / (1. + self.decay * K.cast(self.iterations,*/
/*K.dtype(self.decay))))*/

/*t = K.cast(self.iterations, K.floatx()) + 1*/
/*lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /*/
/*(1. - K.pow(self.beta_1, t)))*/

/*ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]*/
/*vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]*/
/*self.weights = [self.iterations] + ms + vs*/

/*for p, g, m, v in zip(params, grads, ms, vs):*/
/*m_t = (self.beta_1 * m) + (1. - self.beta_1) * g*/
/*v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)*/
/*p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)*/

/*self.updates.append(K.update(m, m_t))*/
/*self.updates.append(K.update(v, v_t))*/
/*new_p = p_t*/

/*# Apply constraints.*/
/*if getattr(p, 'constraint', None) is not None:*/
/*new_p = p.constraint(new_p)*/

/*self.updates.append(K.update(p, new_p))*/
/*return self.updates*/

/*def get_config(self):*/
/*config = {'lr': float(K.get_value(self.lr)),*/
/*'beta_1': float(K.get_value(self.beta_1)),*/
/*'beta_2': float(K.get_value(self.beta_2)),*/
/*'decay': float(K.get_value(self.decay)),*/
/*'epsilon': self.epsilon}*/
/*base_config = super(Adam, self).get_config()*/
/*return dict(list(base_config.items()) + list(config.items()))*/
