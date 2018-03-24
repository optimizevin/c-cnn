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
            tmp = Relu_def(tmp);
            tmp += bias;
            (*(float_t(*)[][nbox_cols])pOut)[ida][jda] = tmp;
        }
    }
}

inline  void conv2d_withlayer(float_t *pneu, uint32_t data_rows, uint32_t data_cols, uint32_t data_batch,
                              struct conv_layer *pconv_layer)
{
    assert(pneu != NULL);
    if(pconv_layer->conv_out == NULL) {
        uint32_t cols = data_cols - pconv_layer->fl_cols + 1;
        uint32_t rows = data_rows - pconv_layer->fl_rows + 1;
        pconv_layer->conv_out = (float_t*)calloc(data_batch * cols * rows * pconv_layer->fl_batch * sizeof(float_t), 1);
        pconv_layer->out_batch = data_batch * pconv_layer->fl_batch;
    }

    float_t (*pfilter)[pconv_layer->fl_rows][pconv_layer->fl_cols] =
        (float_t(*)[pconv_layer->fl_rows][pconv_layer->fl_cols])pconv_layer->filter_core;

    const uint32_t  nbox_rows = data_rows - pconv_layer->fl_rows + 1;
    const uint32_t  nbox_cols = data_cols - pconv_layer->fl_cols + 1;

    pconv_layer->out_rows = nbox_rows;
    pconv_layer->out_cols = nbox_cols;

    float_t (*pout)[nbox_rows][nbox_cols] =
        (float_t(*)[nbox_rows][nbox_cols])pconv_layer->conv_out;


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

    if(ppool_layer->pool_out) {
        free(ppool_layer->pool_out);
        ppool_layer->pool_out =  NULL;
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

    ppool_layer->pool_out = (float_t*)calloc(out_rows * out_cols * batch * sizeof(float_t) , 1);
    for(uint32_t i = 0; i < batch; i++) {
        uint32_t offset = out_rows * out_cols * i ;
        max_pool(pData + offset, data_rows, data_cols, ppool_layer->pl_rows,
                 ppool_layer->pl_cols, stride, ppool_layer->pool_out + offset);
    }
    ppool_layer->out_batch =  batch;

}


struct input_layer* create_inputlayer(const char* pstr, const float_t *pdata, uint32_t cols, uint32_t rows,
                                      const uint32_t label)
{
    assert(pdata != NULL);
    struct input_layer* ret = NULL;
    const uint32_t fullsize = cols * rows ;

    ret = (struct input_layer*)calloc(sizeof(struct input_layer), 1);
    ret->base.laytype = LAY_INPUT;
    strcpy(ret->base.layerName, pstr);
    ret->in_cols = cols;
    ret->in_rows = rows;
    ret->nenum = fullsize;

    ret->pdata = (float_t*)calloc(fullsize * sizeof(float_t), 1);
    ret->value = label;

    uint32_t i = 0;
    uint32_t limit = fullsize - 3;

    for( i = 0; i < limit; i += 4 ) {
        ret->pdata[i] = pdata[i];
        ret->pdata[i + 1] = pdata[i + 1];
        ret->pdata[i + 2] = pdata[i + 2];
        ret->pdata[i + 3] = pdata[i + 3];
    }

    for(; i < fullsize; i++) {
        ret->pdata[i] = pdata[i];
    }

    return ret;
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


struct fc_layer* create_fully_connected_layer(const char*pstr, uint32_t neunum, float_t bias)
{
    struct  fc_layer * ret = (struct fc_layer*)calloc(sizeof(struct fc_layer), 1);
    ret->base.laytype = LAY_FULLYCONNECT;
    strcpy(ret->base.layerName, pstr);
    ret->neunum = neunum;
    if(ret->neu == NULL) {
        ret->neu = (float_t*)calloc(sizeof(float_t) * neunum, 1);
    }
    if(ret->weight == NULL) {
        ret->weight = (float_t*)calloc(sizeof(float_t) * neunum, 1);
    }

    for(uint32_t  i = 0; i < neunum; i++) {
        ret->weight[i] =  generateGaussianNoise(0.5f, 0.8f);
    }
    ret->bias = bias;
    return ret;
}

inline  void fully_connected_data(float_t *pdata, uint32_t data_rows, uint32_t data_cols, uint32_t data_batch,
                                  float_t *pweight, float_t bias, float_t *pout)
{
    assert(pdata != NULL);
    float_t (*pd)[data_rows][data_cols] =
        (float_t(*)[data_rows][data_cols])pdata;


    float_t tmp = 0.f;
    uint32_t step = 0;
    for(uint32_t db = 0; db < data_batch; db++) {
        for(uint32_t i = 0; i < data_rows; i++) {
            for(uint32_t j = 0; j < data_cols; j++) {
                tmp += (*pd)[i][j] * pweight[step];
            }
        }
        pout[step] = Relu_def(tmp + bias);
        tmp = 0.f;
        pd++;
        step++;
    }

}


inline  void fully_connected_fclayer(float_t *pdata, uint32_t data_rows, uint32_t data_cols,
                                     uint32_t data_batch, struct fc_layer *pfc_layer)
{
    assert(pdata != NULL);
    fully_connected_data(pdata, data_rows, data_cols, data_batch,
                         pfc_layer->weight, pfc_layer->bias, pfc_layer->neu);
}

inline  void forward_proc(uint32_t label, struct output_layer * pout)
{
    static int labelarray[10] = {0};
    assert(label < 10 && label >= 0);
    labelarray[label + 1] = 1;

    /*softMax_cross_entropy_with_logits(blabel, b, 2, 3, out);*/
    /*softmax_cross_entropy_with_logits(labelarray,pout->output);*/
    /*y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2*/

    /*cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))*/
    /*train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)*/


    /*cross_entropy = reduece_mean(softmax_cross_entropy_with_logits(label,fc->out);*/
    /*float_t out[2] ;*/
    /*softMax_cross_entropy_with_logits(blabel, b, 2, 3, out);*/
    /*reduece_mean(pfc_layer->out);*/
}

struct dropout_layer* create_dropout_layer(const char*pstr, uint32_t rows, uint32_t cols, uint32_t batch)
{
    struct  dropout_layer * ret = (struct dropout_layer*)calloc(sizeof(struct dropout_layer), 1);
    ret->base.laytype = LAY_DROPOUT;
    strcpy(ret->base.layerName, pstr);
    ret->out_rows = rows;
    ret->out_cols = cols;
    ret->drop_batch = batch;
    ret->drop_out = (float_t*)calloc(cols * rows * batch * sizeof(float_t), 1);
    return ret;
}

struct output_layer* create_output_layer(const char*pstr, uint32_t neunum)
{
    struct  output_layer * ret = (struct output_layer*)calloc(sizeof(struct output_layer), 1);
    ret->base.laytype = LAY_OUTPUT;
    strcpy(ret->base.layerName, pstr);
    ret->size = neunum;
    ret->output = (float_t*)calloc(neunum * sizeof(float_t), 1);
    return ret;
}


inline void dropout_layer(float_t *pdata, uint32_t rows, uint32_t cols, uint32_t batch, struct dropout_layer *pdrop_layer)
{
    /*printf("drop len = %d\n",rows*cols*batch);*/
    /*dropout((const float_t*)pdata, rows * cols * batch, 0.5, pdrop_layer->drop_out);*/
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

void destory_layer(union store_layer *player)
{
    switch(player->pconv_layer->base.laytype) {
    case  LAY_INPUT: {
        if(player->pinput_layer->pdata) {
            free(player->pinput_layer->pdata);
        }
        free(player->pinput_layer);
    }
    break;
    case  LAY_CONV: {
        if(player->pconv_layer->filter_core) {
            free(player->pconv_layer->filter_core);
        }
        free(player->pconv_layer);
    }
    break;
    case  LAY_POOL: {
        if(player->ppool_layer->pool_out) {
            free(player->ppool_layer->pool_out);
        }
        free(player->ppool_layer);
    }
    break;
    case  LAY_FULLYCONNECT: {
        if(player->pfc_layer->neu) {
            free(player->pfc_layer->neu);
        }
        if(player->pfc_layer->weight) {
            free(player->pfc_layer->weight);
        }
        free(player->pfc_layer);
    }
    break;
    case  LAY_DROPOUT: {
        if(player->pdrop_layer->drop_out) {
            free(player->pdrop_layer->drop_out);
        }
        free(player->pdrop_layer);
    }
    break;
    case  LAY_OUTPUT: {
        if(player->pout_layer->output) {
            free(player->pout_layer->output);
        }
        free(player->pout_layer);
    }
    break;

    default:
        break;
    }
}
