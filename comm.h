/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : comm.h
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/

#ifndef  _CNN_COMM
#define  _CNN_COMM

#include <stdint.h>

#define Relu (x) x>0?x:0
//noisyRelu
//leakyRelu
//

#ifndef  CHECK_ARRAY
#define   CHECK_ARRAY
#define   CHECK_ROWS(_) (int)(sizeof(_)/sizeof(_[0]))
#define   CHECK_COLS(_) (int)(sizeof(_[0])/sizeof(_[0][0]))
#endif

inline float sigmoid(const float x);

struct  notelist {
    struct notelist* pNext;
};

struct  data_batch {
    uint32_t  batch;
    uint32_t  in_height;
    uint32_t  in_width;
    float     data[0];
};

inline  float* randf(const uint32_t nsize, const float stddev);
inline  float generateGaussianNoise(const float mean, const float stdDev);
//inline  mutmul

#endif
