/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : nncomm.h
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/

#ifndef  _CNN_COMM
#define  _CNN_COMM

#include <stdint.h>


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

inline  void bias( float *pfloat, const  uint32_t nsize, const float stddev);
inline  float  generateGaussianNoise(const float mean, const float stdDev);
inline  float* randf(const uint32_t nsize, const float stddev);

inline void intMatrixMutiply(const uint32_t *a,const uint32_t *b,uint32_t *c,uint32_t arow,uint32_t acol,uint32_t bcol);
inline void floatMatrixMutiply(const float *a,const float *b,float *c, uint32_t arow, uint32_t acol,uint32_t bcol);
inline float *MatrixAdd(const float *a,const float *b,uint32_t r,uint32_t c);
inline float Relu(const float *pf, uint32_t len);
// maxpool
// reduce_ment
// softmax_
// adamoptimizer

#endif
