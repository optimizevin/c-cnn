/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : comm.c
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/

#include "comm.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>

#define sigmoid_exp fastexp

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

