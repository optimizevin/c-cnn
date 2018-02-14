/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : cnn.h
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
#include <stdio.h>
#include <stdint.h>
#include "nncomm.h"

struct conv_filter_head {
    uint32_t  filter_batch;
    uint32_t  in_height;
    uint32_t  in_width;
	float_t bias;
    float_t   filter_core[0];
};

struct output_block {
    uint32_t  in_height;
    uint32_t  in_width;
    float_t   data[0];
};

struct subsampling{
    uint32_t  in_height;
    uint32_t  in_width;
    float_t   data[0];
};

enum LAYERTYPE {
    LAY_INPUT,
    LAY_CONV,
    LAY_POOL,
    LAY_FULLYCONNECT,
    LAY_OUT
};


struct layer{
    char layerName[128];
	enum LAYERTYPE laytype;
	uint32_t nenum;
    struct conv_filter_head* pconv_filter;
	float_t bias;
	float_t neu[0];
	float_t weight[0];
};


struct conv_filter_head *create_convcore(const uint32_t batch, const uint32_t height,
        const uint32_t width,const float_t mu,const float_t stddev);

inline struct data_batch *conv2d_batch(const struct data_batch * pdatabatch, 
        struct conv_filter_head * filter, const int strides, const int padding);
inline  void conv2d_withonefilter(const float_t *pData, uint32_t data_height, uint32_t data_width,
                    float_t *filter, uint32_t fl_height, uint32_t fl_width, float_t *pOut);

struct layer* makelayer(const char *pstr,uint32_t width,uint32_t height,uint32_t num,
        float_t bias,float_t stddev,enum LAYERTYPE laytype);
//max_pool_2x2;



