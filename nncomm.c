/*
 *   Copyright (C), 2017-1, Tech. Co., Ltd.
 *   File name   : nncomm.c
 *   Author      : vincent
 *   Version     : 0.9
 *   Date        : 2017.6
 *   Description : cnn
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
            /* printf("%d\t",ptr_c[j]); */
        }
        /* printf("\n"); */
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
            /* printf("%f\t",ptr_c[j]); */
        }
        /* printf("\n"); */
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
            /*printf("%d:%d = %f\t",i,j,tmp);*/
            pOut[i][j] =  tmp;
            /*printf("\n");*/
        }
    }
}
