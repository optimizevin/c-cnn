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
#include <assert.h>

#define  NAME_LENGTH  128

struct output_block {
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

struct base_layer {
    char layerName[128];
    enum LAYERTYPE laytype;
};


struct input_layer {
    struct base_layer base;
    uint32_t nenum;
    uint32_t  in_height;
    uint32_t  in_width;
    float_t bias;
    float_t *neu;
    float_t *weight;
};

struct conv_layer {
    struct base_layer base;
    uint32_t  fl_batch;
    uint32_t  fl_height;
    uint32_t  fl_width;
    float_t   bias;
    float_t   *filter_core;
    float_t   *pout;
};

union store_layer {
    struct input_layer* pinputlayer;
    struct conv_layer*  pconvlayer;
};

void logpr(float_t* fd,int dept);

inline  void conv2d_withonefilter(const float_t *pData, uint32_t data_height, uint32_t data_width,
                                  float_t *filter, uint32_t fl_height, uint32_t fl_width, float_t *pOut);
inline  void conv2d_withlayer(float_t *pneu, uint32_t data_height, uint32_t data_width,
                              struct conv_layer *pconv_layer);


struct input_layer* create_inputlayer(const char* pstr, const float_t *pdata, uint32_t width, uint32_t height,
                                      const uint32_t batch, float_t bias, float_t stddev);
struct conv_layer* create_convlayer(const char* pstr, uint32_t width, uint32_t height, uint32_t batch,
                                    float_t bias, float_t stddev);

void destory_convlayer(struct conv_layer* pconv_layer);
void destory_inputlayer(struct input_layer* pinput_layer);


