/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : cnn.c
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
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
#include "cnn.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>

struct conv_filter_head *create_convcore(const uint32_t batch, const uint32_t height,
        const uint32_t width, const float_t mu, const float_t stddev)
{
    const uint32_t size = height * width * batch + sizeof(struct conv_filter_head);
    struct conv_filter_head *pfh = (struct conv_filter_head*)malloc(size);
    memset(pfh, 0x0, size);
    pfh->filter_batch = batch;
    pfh->in_height =  height;
    pfh->in_width = width;
    for(uint32_t  i = 0; i < size; i++) {
        pfh->filter_core[i] =  generateGaussianNoise(mu, stddev);
    }
    /*float_t (*fit)[width] = (float_t(*)[width])pfh->filter_core;*/
    return pfh;

}


inline struct data_batch *conv2d_batch(const struct data_batch * pdatabatch,
                                       struct conv_filter_head * pfilter, const int strides, const int padding)
{
    /*struct  data_batch {*/
    /*uint32_t  batch;*/
    /*uint32_t  in_height;*/
    /*uint32_t  in_width;**/
    /*float_t     data[0];*/
    /*};*/
    /*struct conv_filter_head {*/
    /*uint32_t  filter_batch;*/
    /*uint32_t  in_height;*/
    /*uint32_t  in_width;*/
    /*float_t   filter_core[0];*/
    /*};*/

    /*float_t (*pImg)[pdatabatch->in_height][pdatabatch->in_width] =*/
    /*(float_t(*)[pdatabatch->in_height][pdatabatch->in_width])pdatabatch->data;*/

    /*float_t (*Pimg)[28][28] = &pdatabatch->data;*/

    /*float_t (*pCov)[pfilter->in_height][pfilter->in_width] =*/
    /*(float_t(*)[pfilter->in_height][pfilter->in_width])pfilter->filter_core;*/

    /*float_t tmp  = 0.f;*/
    /*for(uint32_t i = 0; i < pdatabatch->batch; i++) {*/
    /*for(uint32_t j = 0; j < pfilter->filter_batch; i++) {*/
    /*pCov++;*/
    /*tmp = (*pImg)[10][10];*/
    /*}*/
    /*pImg++;*/
    /*}*/

    struct data_batch *pdb =  NULL;
    return pdb;
}


inline  float_t conv2d(float_t *pData, float_t *filter, uint32_t in_height, uint32_t in_width, float *pOut)
{
    float_t (*pImg)[in_height][in_width] =
        (float_t(*)[in_height][in_width])pData;
    float_t (*pfilter)[in_height][in_width] =
        (float_t(*)[in_height][in_width])filter;

    float_t tmp = 0.f;
    for(uint32_t i = 0; i < in_height; i++) {
        for(uint32_t j = 0; j < in_width; j++) {
            tmp += (*pImg)[i][j] * (*pfilter)[i][j];
        }
    }
    return tmp;
}
