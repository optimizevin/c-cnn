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
#include <float.h>

void logpr(float_t* fd, int32_t rows, int32_t cols, int32_t dept)
{
    float_t *pdata = fd + (rows * cols * dept);
    float_t(*p)[][cols] = (float_t(*)[][cols])pdata;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%3.1f ", (*p)[i][j]);
        }
        printf("\n");
    }
}

inline float_t *create_filtercore(const uint32_t batch, const uint32_t cols,
                                  const uint32_t rows,  const float_t stddev)
{
    const uint32_t nsize = cols * rows * batch * sizeof(float_t);
    assert(nsize > 0);
    float_t *pret = (float_t*)calloc(nsize * sizeof(float_t), 1);
    for(uint32_t  i = 0; i < nsize; i++) {
        pret[i] =  generateGaussianNoise(0.5f, 0.8f);
        /*printf("%5.3f \n",pret[i]);*/
    }
    return pret;
}


/*  1, 2, 3, 4,  */           /*  0.1 0.f */
/*  5, 6, 7, 8,  */           /*  0.1 0.f  */
/*  9,10,11,12   */

/* 0.6 0.8 1.0*/
/* 1.4 1.6 1.8*/
inline  void conv2d_withonefilter(const float_t *pData, uint32_t data_rows, uint32_t data_cols,
                                  float_t *filter, uint32_t fl_rows, uint32_t fl_cols, float_t bias, float_t *pOut)
{
    float_t (*pImg)[data_rows][data_cols] =
        (float_t(*)[data_rows][data_cols])pData;
    float_t (*pfilter)[fl_rows][fl_cols] =
        (float_t(*)[fl_rows][fl_cols])filter;


    float_t tmp = 0.f;
    const uint32_t  nbox_rows = data_rows - fl_rows + 1;
    const uint32_t  nbox_cols = data_cols - fl_cols + 1;

    for(uint32_t ida = 0; ida < nbox_rows; ida++) {
        for(uint32_t jda = 0; jda < nbox_cols; jda++) {
            tmp =  0.f;
            for(uint32_t fda = 0; fda < fl_rows; fda++) {
                for(uint32_t fdj = 0; fdj < fl_cols; fdj++) {
                    tmp += (*pImg)[ida + fda][jda + fdj] * (*pfilter)[fda][fdj];
                }
            }
            tmp += bias;
            (*(float_t(*)[][nbox_cols])pOut)[ida][jda] = tmp;
        }
    }
}

inline  void conv2d_withlayer(float_t *pneu, uint32_t data_rows, uint32_t data_cols, uint32_t data_batch,
                              struct conv_layer *pconv_layer)
{
    assert(pneu != NULL);
    if(pconv_layer->pout == NULL) {
        uint32_t cols = data_cols - pconv_layer->fl_cols + 1;
        uint32_t rows = data_rows - pconv_layer->fl_rows + 1;
        pconv_layer->pout = (float_t*)calloc(data_batch * cols * rows * pconv_layer->fl_batch * sizeof(float_t), 1);
        pconv_layer->out_batch = data_batch * pconv_layer->fl_batch;
    }

    float_t (*pfilter)[pconv_layer->fl_rows][pconv_layer->fl_cols] =
        (float_t(*)[pconv_layer->fl_rows][pconv_layer->fl_cols])pconv_layer->filter_core;

    const uint32_t  nbox_rows = data_rows - pconv_layer->fl_rows + 1;
    const uint32_t  nbox_cols = data_cols - pconv_layer->fl_cols + 1;

    pconv_layer->out_rows = nbox_rows;
    pconv_layer->out_cols = nbox_cols;

    float_t (*pout)[nbox_rows][nbox_cols] =
        (float_t(*)[nbox_rows][nbox_cols])pconv_layer->pout;


    float_t *pdata = pneu;
    uint32_t step = data_rows * data_cols;
    float_t (*pStepfilter)[pconv_layer->fl_rows][pconv_layer->fl_cols] = pfilter;
    for(uint32_t i = 0; i < data_batch; i++) {
        for(uint32_t j = 0; j < pconv_layer->fl_batch; j++) {
            conv2d_withonefilter(pdata, data_rows, data_cols, (float_t*)pStepfilter,
                                 pconv_layer->fl_rows, pconv_layer->fl_cols, pconv_layer->bias, (float_t*)pout);
            pStepfilter++;
            pout++;
        }
        pStepfilter = pfilter;
        pdata += step;
    }
}


//Max Pool
inline void pool_withlayer(const float_t*pData, uint32_t data_rows, uint32_t data_cols, uint32_t batch,
                           struct pool_layer *ppool_layer, uint32_t  stride)
{
    assert(ppool_layer);

    if(ppool_layer->poolout) {
        free(ppool_layer->poolout);
        ppool_layer->poolout =  NULL;
    }
    uint32_t  out_rows = 0;
    uint32_t  out_cols = 0;
    for(uint32_t step = 0; step <= (data_rows - stride) ; step += stride) {
        out_rows++;
    }
    for(uint32_t step = 0; step <= (data_cols - stride) ; step += stride) {
        out_cols++;
    }

    ppool_layer->out_rows = out_rows;
    ppool_layer->out_cols = out_cols;

    ppool_layer->poolout = (float_t*)calloc(out_rows * out_cols * batch * sizeof(float_t) , 1);
    for(uint32_t i = 0; i < batch; i++) {
        uint32_t offset = out_rows * out_cols * i ;
        max_pool(pData + offset, data_rows, data_cols, ppool_layer->pl_rows,
                 ppool_layer->pl_cols, stride, ppool_layer->poolout + offset);
    }
    ppool_layer->pl_batch =  batch;

}


struct input_layer* create_inputlayer(const char* pstr, const float_t *pdata, uint32_t cols, uint32_t rows,
                                      const uint32_t batch, float_t bias, float_t stddev)
{
    assert(pdata != NULL);
    struct input_layer* ret = NULL;
    const uint32_t fullsize = cols * rows * batch;

    ret = (struct input_layer*)calloc(sizeof(struct input_layer), 1);
    ret->base.laytype = LAY_INPUT;
    strcpy(ret->base.layerName, pstr);
    ret->in_cols = cols;
    ret->in_rows = rows;
    ret->bias = bias;
    ret->nenum = fullsize;

    ret->neu = (float_t*)calloc(fullsize * sizeof(float_t), 1);
    for(uint32_t i = 0; i < fullsize; i++) {
        ret->neu[i] = pdata[i];
    }

    ret->weight = (float_t*)calloc(fullsize * sizeof(float_t), 1);
    for(uint32_t i = 0; i < fullsize; i++) {
        ret->weight[i] = generateGaussianNoise(0.f, stddev);
    }
    return ret;
}

void destory_inputlayer(struct input_layer* pinput_layer)
{
    if(pinput_layer->neu) {
        free(pinput_layer->neu);
    }
    if(pinput_layer->weight) {
        free(pinput_layer->weight);
    }
    free(pinput_layer);
}


struct conv_layer* create_convlayer(const char* pstr, uint32_t cols, uint32_t rows, uint32_t batch,
                                    float_t bias, float_t stddev)
{
    struct conv_layer* ret = (struct conv_layer*)calloc(sizeof(struct conv_layer), 1);
    ret->base.laytype = LAY_CONV;
    strcpy(ret->base.layerName, pstr);
    ret->fl_rows = rows;
    ret->fl_cols = cols;
    ret->fl_batch = batch;
    ret->bias = bias;
    ret->filter_core = create_filtercore(batch, cols, rows, stddev);

    return ret;
}

void destory_convlayer(struct conv_layer* pconv_layer)
{
    if(pconv_layer->filter_core) {
        free(pconv_layer->filter_core);
    }
    free(pconv_layer);
}

struct pool_layer* create_poollayer(const char* pstr, uint32_t cols, uint32_t rows)
{
    struct pool_layer* ret = (struct pool_layer*)calloc(sizeof(struct pool_layer), 1);
    ret->base.laytype = LAY_POOL;
    strcpy(ret->base.layerName, pstr);
    ret->pl_rows = rows;
    ret->pl_cols = cols;
    return ret;
}

void destory_poollayer(struct pool_layer* pool_layer)
{
    if(pool_layer->poolout) {
        free(pool_layer->poolout);
    }
    free(pool_layer);
}

