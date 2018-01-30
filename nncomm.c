/*
 *   Copyright (C), 2017-1
 *   File name   : nncomm.c
 *   Author      : vincent
 *   Version     : 0.9
 *   Date        : 2017.6
 *   Description : cnn
 *   GitHub      : https://github.com/optimizevin
 */


/***************************************************************************

   Vincent ,
   GitHub      : https://github.com/optimizevin

   Copyright (c) 2017 - .  All rights reserved.

   This code is licensed under the MIT License.  See the FindCUDA.cmake script
   for the text of the license.

  The MIT License

  License for the specific language governing rights and limitations under
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.

***************************************************************************/

#include "nncomm.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>

#define sigmoid_exp fastexp
#define Relu_def(x) x>0?x:0
//noisyRelu
//leakyRelu
//

inline float_t fastexp(float_t x)
{
    /* e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) + (x^4 / 4!) */
    float_t sum = 1 + x;
    float_t n = x;
    float_t d = 1;
    float_t i;

    for(i = 2; i < 100; i++) {
        n *= x;
        d *= i;
        sum += n / d;
    }

    return sum;
}


inline float_t generateGaussianNoise(const float_t mean, const float_t stdDev)
{
    static int hasSpare = 0;
    static float_t spare;

    if(hasSpare) {
        hasSpare = 0;
        return mean + stdDev * spare;
    }

    hasSpare = 1;
    static float_t u, v, s;
    do {
        u = (rand() / ((float_t) RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((float_t) RAND_MAX)) * 2.0 - 1.0;
        s = u * u + v * v;
    } while( (s >= 1.0) || (s == 0.0) );

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + stdDev * u * s;
}

inline float_t sigmoid(const float_t x)
{
    return 1.00 / (1 + sigmoid_exp(0 - x));
}

inline  float_t* randf(const uint32_t nsize, const float_t stddev)
{
    float_t *ret = (float_t*)calloc(nsize, sizeof(float_t));
    for(uint32_t i = 0; i < nsize; i++) {
        ret[i] = generateGaussianNoise(0.f, stddev);
    }

    return ret;
}

inline  void bias(float_t *pfloat_t, const  uint32_t nsize, const float_t stddev)
{
    for(uint32_t i = 0; i < nsize; i++) {
        pfloat_t[i] += stddev;
    }
}


inline float_t Relu(const float_t *pf, uint32_t len)
{
    float_t t = 0.f;
    for(uint32_t i = 0; i < len; i++) {
        float_t  m = Relu_def(pf[i]);
        t =  t > m ? t : m;
    }

    return  t;
}

inline void Dropout(const float_t *src, const uint32_t len, float_t keep_prob, float_t *out)
{
    for(int i = 0; i < len; i++) {
        float_t tmp = (rand() / ((float_t)RAND_MAX)) ;
        (tmp > (1 - keep_prob)) ? (out[i] = src[i]) : (out[i] = 0);
    }
}


inline float_t *MatrixAdd(const float_t *a, const float_t *b, uint32_t r, uint32_t c)
{
    float_t * pret = (float_t*)calloc(r * c, sizeof(float_t));
    for(uint32_t i = 0; i < r; i++) {
        for(uint32_t j = 0; j < c; j++) {
            ((float_t (*)[c])pret)[i][j] = ((float_t (*)[c])(a))[i][j] + ((float_t (*)[c])(b))[i][j];
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

inline void float_tMatrixMutiply(const float_t *a, const float_t *b, float_t *c, uint32_t arow, uint32_t acol, uint32_t bcol)
{
    for(uint32_t i = 0; i < arow; i++) {
        float_t *ptr_c = c + i * bcol;
        float_t *ptr_a = (float_t*)a + i * acol;
        for(uint32_t j = 0; j < bcol; j++) {
            for(uint32_t k = 0; k < acol; k++) {
                ptr_c[j] += ptr_a[k] * b[k * bcol + j];
            }
        }
    }
}


inline void max_pool(float_t *src, uint32_t rows, uint32_t cols, uint32_t pool_size, float_t*pout)
{
    float_t(*pSrc)[cols] = (float_t(*)[cols])src;
    float_t(*pOut)[cols - pool_size + 1] = (float_t(*)[cols - pool_size + 1])pout;

    for(uint32_t i = 0; i <= (rows - pool_size); i++) {
        for(uint32_t j = 0; j <= (cols - pool_size); j++) {
            float_t tmp = 0.f;
            for(uint32_t ip = 0; ip < pool_size; ip++) {
                for(uint32_t jp = 0; jp < pool_size; jp++) {
                    tmp =  MAX(pSrc[ip + i][jp + j], tmp);
                }
            }
            pOut[i][j] =  tmp;
        }
    }
}


inline float_t reduce_ment(const float_t *src, const uint32_t len)
{
    float_t ret = 0.f;
    for(uint32_t i = 0 ; i < len; i++) {
        ret += src[i];
    }
    ret /= len;
    return ret;
}


inline void foreach_log(float_t *src, const uint32_t len)
{
    for(uint32_t i = 0 ; i < len; i++) {
        src[i] = log(src[i]);
    }
}


inline void softMax(float_t *src, uint32_t rows, uint32_t cols)
{
    for (uint32_t i = 0; i < rows; ++i) {
        float_t max = 0.0;
        float_t sum = 0.0;
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

/************************************************
    Cross-entropy cost function
************************************************/
inline void softMax_cross_entropy_with_logits(const float_t *labels, const float_t *logits,
        uint32_t rows, uint32_t cols, float_t *pOut)
{
    float_t tmp[rows * cols];
    memcpy(tmp, logits, rows * cols * sizeof(float_t));
    softMax(tmp, rows, cols);
    const  uint32_t bsize = sizeof(tmp) / sizeof(tmp[0]);
    foreach_log(tmp, bsize);

    float_t bout[bsize] ;
    /*memset(bout, 0x0, bsize);*/
    for(uint32_t i = 0; i < bsize; i++) {
        bout[i] = tmp[i] * labels[i];
    }

    float_t(*pBout)[cols] = (float_t(*)[cols])bout;
    for(uint32_t i = 0; i < rows; i++) {
        float_t  t = 0.f;
        for(uint32_t j = 0; j < cols; j++) {
            t += pBout[i][j];
        }
        pOut[i] = -t;
    }

}




/************************************************
  "SGD & momentum
       lr: float_t = 0.01 Learning rate.
       momentum: float_t =  0.f
       decay: float_t = 0. Learning rate decay over each update.
       nesterov: bool =  false
************************************************/
inline void SGD_Momentum(const float_t *W,const uint32_t len,const float_t lr, const float_t momentum,
                         const float_t decay, const int nesterov)
{
   for(uint32_t i=0;i<len;i++){
   } 
}




/************************************************
   "Adam optimizer.

   Default parameters follow those provided in the original paper.

       lr: float_t >= 0. Learning rate.
       beta_1: float_t, 0 < beta < 1. Generally close to 1.
       beta_2: float_t, 0 < beta < 1. Generally close to 1.
       epsilon: float_t >= 0. Fuzz factor.
       decay: float_t >= 0. Learning rate decay over each update.

   # References
   - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
     learning_rate=0.001,
     beta1=0.9, beta2=0.999, epsilon=1e-08,
     use_locking=False,

************************************************/
inline  void AdamOptimizer(const float_t lr, const float_t beta_1, const float_t beta_2,
                           const float_t epsilon, const float_t decay)
{

}


inline  void core_forback()
{
    uint32_t  max_epoc = 2000;
    const float threshold = 0.05;
    for(uint32_t  istep = 0;i<max_epoc;i++){
        for(uint32_t count_sample = 0 ;count_sample < m;count_sample++){
            /*ai to xi*/
            for(ll =2 ;ll<L;ll++){
                for_forward(ai);
            }
            const(bp_ret);
            for(ll =L ;ll>2;ll--){
                for_back(ai);
            }
            update(ll[wl,wbl,bl];
            if(ll[wl,wbl,bl] <threshold)
                goto exit;
        }
    }
exit:
}


