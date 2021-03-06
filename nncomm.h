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

#define SAFEFREE(_) if(_){free(_);}

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)>(b))?(b):(a))
#define Relu_def(x) x>0?x:0

inline float_t sigmoid(const float_t x);
inline float_t diff_sigmoid(float_t y);


typedef  void(*pFun_Pooll)(float_t *, uint32_t , uint32_t , uint32_t , float_t*);

inline void padding(float_t* p,uint32_t rows,uint32_t cols,uint32_t step,float_t *pout);
 //Min-Max normalization 
inline void MinMax(float_t *pdata,uint32_t rows,uint32_t cols);
inline void MinMax_log(float_t *pdata, uint32_t rows, uint32_t cols);

inline  void bias( float_t *pfloat, const  uint32_t nsize, const float_t stddev);
inline  float_t  generateGaussianNoise(const float_t mean, const float_t stdDev);
inline  float_t* randf(const uint32_t nsize, const float_t stddev);

inline void intMatrixMutiply(const uint32_t *a, const uint32_t *b, uint32_t *c,
        uint32_t arow, uint32_t acol, uint32_t bcol);
//Hadamard mutiply
inline void float_tMatrixMutiply(const float_t *a, const float_t *b, float_t *c,
        uint32_t arow, uint32_t acol, uint32_t bcol);
inline float_t *MatrixAdd(const float_t *a, const float_t *b, uint32_t r, uint32_t c);

inline void max_pool(const float_t*pData, uint32_t data_rows, uint32_t data_cols,
                     uint32_t pl_rows , uint32_t pl_cols, uint32_t stride, float_t * pout);
inline void ave_pool(const float_t*pData, uint32_t data_rows, uint32_t data_cols,
                     uint32_t pl_rows , uint32_t pl_cols, uint32_t stride, float_t * pout);

inline void subsampling_fun(float_t *src, uint32_t rows, uint32_t cols, uint32_t pool_size, pFun_Pooll pfunpool,float_t*pout);

inline float_t Relu(const float_t *pf, uint32_t len);
inline void dropout(const float_t *src, const uint32_t len, float_t keep_prob, float_t *out);
inline float_t reduce_mean(const float_t *src, const uint32_t len);
inline void softMax_cross(float_t *src, uint32_t rows, uint32_t cols);
inline void softMax(float_t *src, uint32_t size);
inline float_t softMax_diff(float_t *src, uint32_t pos);
inline void softMax_cross_entropy_with_logits(const uint32_t labels, float_t *logits, const uint32_t blen, float_t *pOut);

inline void foreach_log(float_t *src, const uint32_t len,const float_t bias);

// adamoptimizer
// tf.equal  相等
// tf.argmax  返回最大值的最表
//

#endif
