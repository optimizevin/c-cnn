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
    const uint32_t nbatch = height * width * batch;
    const uint32_t size = nbatch * sizeof(float_t) + sizeof(struct conv_filter_head);
    struct conv_filter_head *pfh = (struct conv_filter_head*)malloc(size);
    memset(pfh, 0x0, size);
    pfh->filter_batch = batch;
    pfh->in_height =  height;
    pfh->in_width = width;
    for(uint32_t  i = 0; i < nbatch; i++) {
        pfh->filter_core[i] =  generateGaussianNoise(mu, stddev);
    }
    return pfh;

}

inline float_t *create_filtercore(const uint32_t batch, const uint32_t width,
                                  const uint32_t height,  const float_t stddev)
{
    const uint32_t nsize = width * height * batch * sizeof(float_t);
    assert(nsize > 0);
    float_t *pret = (float_t*)calloc(nsize*sizeof(float_t), 1);
    for(uint32_t  i = 0; i < nsize; i++) {
        pret[i] =  generateGaussianNoise(0.f, stddev);
    }
    return pret;
}


/*  1, 2, 3, 4,  */           /*  0.1 0.f */
/*  5, 6, 7, 8,  */           /*  0.1 0.f  */
/*  9,10,11,12   */

/* 0.6 0.8 1.0*/
/* 1.4 1.6 1.8*/
inline  void conv2d_withonefilter(const float_t *pData, uint32_t data_height, uint32_t data_width,
                                  float_t *filter, uint32_t fl_height, uint32_t fl_width, float_t *pOut)
{
    float_t (*pImg)[data_height][data_width] =
        (float_t(*)[data_height][data_width])pData;
    float_t (*pfilter)[fl_height][fl_width] =
        (float_t(*)[fl_height][fl_width])filter;

    float_t tmp = 0.f;
    const uint32_t  nbox_height = data_height - fl_height + 1;
    const uint32_t  nbox_width = data_width - fl_width + 1;

    for(uint32_t ida = 0; ida < nbox_height; ida++) {
        for(uint32_t jda = 0; jda < nbox_width; jda++) {
            tmp =  0.f;
            for(uint32_t fda = 0; fda < fl_height; fda++) {
                for(uint32_t fdj = 0; fdj < fl_width; fdj++) {
                    tmp += (*pImg)[ida + fda][jda + fdj] * (*pfilter)[fda][fdj];
                }
            }
            (*(float_t(*)[][nbox_width])pOut)[ida][jda] = tmp;
        }
    }
}

inline  void conv2d_withlayer(struct layer* player, float_t *pOut)
{
    /*uint32_t  data_width =  player->in_width;*/
    /*uint32_t  data_height = player->in_height;*/

    /*uint32_t  fl_width =  player->pconv_filter->in_width;*/
    /*uint32_t  fl_height = player->pconv_filter->in_height;*/

    /*float_t *pData = player->neu;*/
    /*float_t *filter = player->pconv_filter->filter_core;*/

    /*float_t (*pImg)[data_height][data_width] =*/
    /*(float_t(*)[data_height][data_width])pData;*/
    /*float_t (*pfilter)[fl_height][fl_width] =*/
    /*(float_t(*)[fl_height][fl_width])filter;*/

    /*float_t tmp = 0.f;*/
    /*const uint32_t  nbox_height = data_height - fl_height + 1;*/
    /*const uint32_t  nbox_width = data_width - fl_width + 1;*/

    /*for(uint32_t ida = 0; ida < nbox_height; ida++) {*/
    /*for(uint32_t jda = 0; jda < nbox_width; jda++) {*/
    /*tmp =  0.f;*/
    /*for(uint32_t fda = 0; fda < fl_height; fda++) {*/
    /*for(uint32_t fdj = 0; fdj < fl_width; fdj++) {*/
    /*tmp += (*pImg)[ida + fda][jda + fdj] * (*pfilter)[fda][fdj];*/
    /*}*/
    /*}*/
    /*(*(float_t(*)[][nbox_width])pOut)[ida][jda] = tmp;*/
    /*}*/
    /*}*/
}

struct input_layer* create_inputlayer(const char* pstr, const float_t *pdata, uint32_t width, uint32_t height,
                                      const uint32_t batch, float_t bias, float_t stddev)
{
    assert(pdata != NULL);
    struct input_layer* ret = NULL;
    const uint32_t fullsize = width * height * batch;

    ret = (struct input_layer*)calloc(sizeof(ret),1);
    ret->base.laytype = LAY_INPUT;
    strcpy(ret->base.layerName, pstr);
    ret->in_width = width;
    ret->in_height = height;
    ret->bias = bias;
    ret->nenum = fullsize;
    ret->neu = (float_t*)calloc(fullsize*sizeof(float_t), 1);
    memcpy(ret->neu, pdata, fullsize);
    ret->weight = (float_t*)calloc(fullsize*sizeof(float_t), 1);
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


struct conv_layer* create_convlayer(const char* pstr, uint32_t width, uint32_t height, uint32_t batch,
                                    float_t bias, float_t stddev)
{
    struct conv_layer* ret = (struct conv_layer*)calloc(sizeof(ret),1);
    ret->fl_width = width;
    ret->fl_height = height;
    ret->filter_batch = batch;
    ret->bias = bias;
    ret->filter_core = create_filtercore(batch, width, height, stddev);
    return ret;
}

void destory_convlayer(struct conv_layer* pconv_layer)
{
    if(pconv_layer->filter_core) {
        free(pconv_layer->filter_core);
    }
    free(pconv_layer);
}

