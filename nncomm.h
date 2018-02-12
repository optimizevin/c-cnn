/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : nncomm.h
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

#pragma once
#ifndef  _CNN_COMM
#define  _CNN_COMM

#include <stdint.h>
#include <math.h>

#ifndef  CHECK_ARRAY
#define   CHECK_ARRAY
#define   CHECK_ROWS(_) (int)(sizeof(_)/sizeof(_[0]))
#define   CHECK_COLS(_) (int)(sizeof(_[0])/sizeof(_[0][0]))
#endif


#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)>(b))?(b):(a))

inline float_t sigmoid(const float_t x);

struct  notelist {
    struct notelist* pNext;
};


struct  data_batch {
    uint32_t  batch;
    uint32_t  in_height;
    uint32_t  in_width;
    float_t     data[0];
};

inline  void bias( float_t *pfloat_t, const  uint32_t nsize, const float_t stddev);
inline  float_t  generateGaussianNoise(const float_t mean, const float_t stdDev);
inline  float_t* randf(const uint32_t nsize, const float_t stddev);

inline void intMatrixMutiply(const uint32_t *a, const uint32_t *b, uint32_t *c,
        uint32_t arow, uint32_t acol, uint32_t bcol);
inline void float_tMatrixMutiply(const float_t *a, const float_t *b, float_t *c,
        uint32_t arow, uint32_t acol, uint32_t bcol);
inline float_t *MatrixAdd(const float_t *a, const float_t *b, uint32_t r, uint32_t c);
inline void max_pool(float_t *src, uint32_t rows, uint32_t cols, uint32_t pool_size, float_t*pout);
inline float_t Relu(const float_t *pf, uint32_t len);
inline void Dropout(const float_t *src, const uint32_t len, float_t keep_prob, float_t *out);
inline float_t reduce_ment(const float_t *src, const uint32_t len);
inline void softMax(float_t *src, uint32_t rows, uint32_t cols);
inline void softMax_cross_entropy_with_logits(const float_t *labels, const float_t *logits, 
        uint32_t rows, uint32_t cols,float_t *pOut);

inline void foreach_log(float_t *src, const uint32_t len);
inline void SGD_Momentum(const float_t *W,const uint32_t len,const float_t lr, const float_t momentum,
                         const float_t decay, const int nesterov);
inline  void AdamOptimizer(const float_t lr,const float_t beta_1,const float_t beta_2,
        const float_t epsilon,const float_t decay);

// adamoptimizer
// tf.equal  相等
// tf.argmax  返回最大值的最表
//

#endif
