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


void logpr(float_t* fd, int32_t rows, int32_t cols, int32_t epoch)
{
    float_t *pdata = fd + (rows * cols * epoch);
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
            /*tmp = tan(tmp);*/
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
    assert(ppool_layer != NULL);

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


struct input_layer* create_inputlayer(const char* pstr, uint32_t cols, uint32_t rows)
{
    struct input_layer* ret = NULL;
    const uint32_t fullsize = cols * rows ;

    ret = (struct input_layer*)calloc(sizeof(struct input_layer), 1);
    ret->base.laytype = LAY_INPUT;
    strcpy(ret->base.layerName, pstr);
    ret->in_cols = cols;
    ret->in_rows = rows;

    ret->pdata = (float_t*)calloc(fullsize * sizeof(float_t), 1);

    return ret;
}

inline void load_inputlayer(struct input_layer *pinput, const float_t *pdata, const uint32_t label)
{
    assert(pinput != NULL);
    assert(pdata != NULL);

    const uint32_t fullsize = pinput->in_cols * pinput->in_rows ;
    uint32_t i = 0;
    uint32_t limit = fullsize - 3;

    for( i = 0; i < limit; i += 4 ) {
        pinput->pdata[i] = pdata[i];
        pinput->pdata[i + 1] = pdata[i + 1];
        pinput->pdata[i + 2] = pdata[i + 2];
        pinput->pdata[i + 3] = pdata[i + 3];
    }

    for(; i < fullsize; i++) {
        pinput->pdata[i] = pdata[i];
    }
    pinput->label = label;

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


struct fc_layer* create_fully_connected_layer(const char*pstr, uint32_t neunum, uint32_t weightsize_ofoneneu, float_t bias)
{
    struct  fc_layer * ret = (struct fc_layer*)calloc(sizeof(struct fc_layer), 1);
    ret->base.laytype = LAY_FULLYCONNECT;
    strcpy(ret->base.layerName, pstr);
    ret->neunum = neunum;
    ret->weightsize_ofoneneu = weightsize_ofoneneu;
    if(ret->neu == NULL) {
        ret->neu = (float_t*)calloc(sizeof(float_t) * neunum , 1);
    }
    const uint32_t wsize = neunum * weightsize_ofoneneu;
    if(ret->weight == NULL) {
        ret->weight = (float_t*)calloc(sizeof(float_t) * wsize , 1);
    }

    for(uint32_t  i = 0; i < wsize; i++) {
        ret->weight[i] =  generateGaussianNoise(0.5f, 0.8f);
    }
    ret->bias = (float_t*)calloc(sizeof(float_t), neunum);
    for(uint32_t  i = 0; i < neunum; i++) {
        ret->bias[i] = bias;
    }
    return ret;
}

inline  void fully_connected_updata(float_t *pdata, uint32_t data_rows, uint32_t data_cols, uint32_t data_batch,
                                    float_t *pweight, float_t bias, float_t *pout)
{
    assert(pdata != NULL);
    assert(pweight != NULL);
    float_t (*pd)[data_rows][data_cols] =
        (float_t(*)[data_rows][data_cols])pdata;

    float_t tmp = 0.f;
    uint32_t step = 0;
    for(uint32_t db = 0; db < data_batch; db++) {
        for(uint32_t i = 0; i < data_rows; i++) {
            for(uint32_t j = 0; j < data_cols; j++) {
                tmp += (*pd)[i][j] * pweight[step];
                step++;
            }
        }
    }
    *pout = tan(tmp + bias);
    /**pout = Relu_def(tmp + bias);*/
    /**pout = sigmoid(tmp + bias);*/
    /**pout = 777.7f;*/
    /*printf("step:%d\tpout:%8.4f\n", step,*pout);*/
}


inline  void fully_connected_fclayer(float_t *pdata, uint32_t data_rows, uint32_t data_cols,
                                     uint32_t data_batch, struct fc_layer *pfc_layer)
{
    assert(pdata != NULL);
    /*printf("neunum:%d\n", pfc_layer->neunum);*/
    /*printf("rows:%d\n", data_rows);*/
    /*printf("cols:%d\n", data_cols);*/
    /*printf("data_batch:%d\n", data_batch);*/
    for(uint32_t i = 0; i < pfc_layer->neunum; i++) {
        /*printf("fc bias:%8.8f\n", pfc_layer->bias[i]);*/
        fully_connected_updata(pdata, data_rows, data_cols, data_batch,
                               &pfc_layer->weight[i * pfc_layer->weightsize_ofoneneu] , pfc_layer->bias[i], &pfc_layer->neu[i]);
    }
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

inline void dropout_layer(float_t *pdata, uint32_t rows, uint32_t cols, uint32_t batch, struct dropout_layer *pdrop_layer)
{
    dropout((const float_t*)pdata, rows * cols * batch, 0.5, pdrop_layer->drop_out);
}

struct output_layer* create_output_layer(const char*pstr, uint32_t classnum)
{
    struct  output_layer * ret = (struct output_layer*)calloc(sizeof(struct output_layer), 1);
    ret->base.laytype = LAY_OUTPUT;
    strcpy(ret->base.layerName, pstr);
    ret->classnum = classnum;
    ret->output = 0.f;
    ret->input = (float_t*)calloc(classnum * sizeof(float_t), 1);
    return ret;
}

inline void output_epoch( struct fc_layer *pfc_layer, struct output_layer *pout_layer, uint32_t label, uint32_t len,
                          float_t *pdata, uint32_t data_rows, uint32_t data_cols, uint32_t data_batch)
{
    assert(pfc_layer != NULL);
    assert(pout_layer != NULL);


    float_t derivative[len] ;
    for(int loop = 0; loop < 500; loop++) {
        memset(derivative, 0x0, len);
        softMax_cross_entropy_with_logits(label, pfc_layer->neu, pfc_layer->neunum, &derivative[label]);
        float_t err = sigmoid(derivative[label] - 1);

        printf("loop:%d\tcost:%8.3f\terr:%8.3f\n", loop, derivative[label], sigmoid(derivative[label] - 1));
        /*printf("loop:%d\terr:%8.3f\n",loop,sigmoid(derivative[label]-1));*/
        /*printf("loop:%d\tderivative:%8.4f\n", loop, derivative[label]);*/
        /*float_t etotal = 0.f;*/
        /*for(int i = 0; i < pfc_layer->neunum; i++) {*/
        /*etotal +=*/
        /*}*/
        float theta = 0.05;
        for(int i = 0; i < pfc_layer->neunum; i++) {
            for(int j = 0; j < pfc_layer->weightsize_ofoneneu; j++) {
                float w = pfc_layer->weight[i * pfc_layer->weightsize_ofoneneu + j];
                w = pfc_layer->weight[i * pfc_layer->weightsize_ofoneneu + j];
                w -= theta * err * derivative[label];
                pfc_layer->weight[i * pfc_layer->weightsize_ofoneneu + j] = w;
                /*printf("%4dw:%8.8f\n",j,w);*/
                /*printf("i:%d  j:%d\tw:%8.3f\tdw:%8.3f\tout:%8.6f\n",i,j,w,diff/w,pfc_layer->weight[i*pfc_layer->weightsize_ofoneneu+j]);*/
            }
            /*printf("weight:%8.6f\n",pfc_layer->weight[0]);*/
            /*printf("%4d bias:%8.6f\n",i,pfc_layer->bias[i]);*/
            /*pfc_layer->bias[i] = pfc_layer->bias[i]  - theta * (derivative[i] / pfc_layer->bias[i] );*/
            /*float_t b = pfc_layer->bias[i];*/
            /*[>printf("loop:%d\tbias:%8.4f\t pfc:%8.4f\tdiff:%8.4f\n",i,b,pfc_layer->bias[i],derivative);<]*/
            /*b = pfc_layer->bias[i]  - theta * (derivative[i] / pfc_layer->bias[i]) ;*/
            /*if(isnan(b)) {*/
            /*printf("theta:%8.8f\tdiff:%8.8f\n", theta, derivative[i]);*/
            /*}*/
        }

        for(uint32_t i = 0; i < pfc_layer->neunum; i++) {
            fully_connected_updata(pdata, data_rows, data_cols, data_batch,
                                   &pfc_layer->weight[i * pfc_layer->weightsize_ofoneneu] , pfc_layer->bias[i], &pfc_layer->neu[i]);
        }
    }

    /*printf("-----------------------------\n");*/
//SGD moment
    /*pout_layer->output;*/
}


/*def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):*/
/*if test_data:*/
/*n_test = len(test_data)#有多少个测试集*/
/*n = len(training_data)*/
/*for j in xrange(epochs):*/
/*random.shuffle(training_data)*/
/*mini_batches = [*/
/*training_data[k:k+mini_batch_size] */
/*for k in xrange(0,n,mini_batch_size)]*/
/*for mini_batch in mini_batches:*/
/*self.update_mini_batch(mini_batch, eta)*/
/*if test_data:*/
/*print "Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data),n_test)*/
/*else:*/
/*print "Epoch {0} complete".format(j)    */

/*def update_mini_batch(self, mini_batch,eta): */
/*nabla_b = [np.zeros(b.shape) for b in self.biases]*/
/*nabla_w = [np.zeros(w.shape) for w in self.weights]*/
/*for x,y in mini_batch:*/
/*delta_nabla_b, delta_nable_w = self.backprop(x,y)#目标函数对b和w的偏导数*/
/*nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]*/
/*nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]#累加b和w*/
/*#最终更新权重为*/
/*self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]*/
/*self.baises = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.baises, nabla_b)]*/



inline void output_epoch_bias( struct fc_layer *pfc_layer, struct output_layer *pout_layer, uint32_t label, uint32_t len,
                               float_t *pdata, uint32_t data_rows, uint32_t data_cols, uint32_t data_batch)
{
    /*
    float theta = 0.25;
    float  x = 5.f;
    for(int loop = 0; loop < 10; loop++) {
        float grad = 2 * x - 2;
        x -= grad * theta;
        printf("loop:%d\tgard:%8.3f\tx:%8.3f\n", loop, grad, x);
        if (fabs(grad) < 1e-6)
            break;
    }
    return;
    //*/
    /*
    w = [2,-3,-3] # 我们随机给定一组权重
    x = [-1, -2]
    
    # 前向传播
    dot = w[0]*x[0] + w[1]*x[1] + w[2]
    f = 1.0 / (1 + math.exp(-dot)) # sigmoid函数
    
    # 反向传播经过该sigmoid神经元
    ddot = (1 - f) * f # sigmoid函数偏导
    dx = [w[0] * ddot, w[1] * ddot] # 在x这条路径上的反向传播
    dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot] # 在w这条路径上的反向传播
    # yes！
    //*/

    assert(pfc_layer != NULL);
    assert(pout_layer != NULL);

    float rate = 0.01f;
    float d[] = {10.f, 20.f, 30.f};
    float y[] = {0.f, 0.f, 1.f};
    float w[] = {0.8f, 0.6f, 0.4f};
    float dd[3] = {0.f};
    float neu[3] = {0};
    memcpy(dd, d, sizeof(float) * 3);

    float out = 0.f;
    float bias[3] = {10.f, 10.f, 10.f};
    /*softMax_cross_entropy_with_logits(2, d, 3, &out);*/
    /*printf("out: %4.3f\n", out);*/

    for(int loop = 0; loop < 30; loop++) {
        printf("dd1:%4.3f\tdd2:%4.3f\tdd3:%4.3f\n",dd[0],dd[1],dd[2]);
        softMax(dd, 3);
        w[0] = w[0] - rate * (dd[0] - y[0]) * d[0];
        w[1] = w[1] - rate * (dd[1] - y[1]) * d[1];
        w[2] = w[2] - rate * (dd[2] - y[2]) * d[2];
        bias[0] = bias[0] - rate * (0);
        bias[1] = bias[1] - rate * (0);
        bias[2] = bias[2] - rate * (1);
        /*printf("w1:%4.3f\tw2:%4.3f\tw3:%4.3f\n", w[0],w[1],w[2]);*/
        /*printf("bias1:%4.3f\tbias2:%4.3f\tbias3:%4.3f\n", bias[0],bias[1],bias[2]);*/

        for(int j = 0; j < 3; j++) {
            for(int i = 0; i < 3; i++) {
                neu[j] = tan(w[i] * d[i] + bias[i]);
            }
            /*printf("neu[%d]:%4.3f\t",j,neu[j]);*/
        }
        /*printf("\n");*/
        float neutmp[3]={0};
        memcpy(&neutmp,neu,sizeof(float)*3);
        float tmp = 0.f;
        softMax_cross_entropy_with_logits(2, neutmp, 3, &tmp);
        /*printf("tmp:%4.3f\n",tmp);*/


        /*softMax_cross_entropy_with_logits(2, d, 3, &tmp);*/
        /*printf("out: %4.3f\tbias:%4.3f\td1:%4.3f\td2:%4.3f\td3:%4.3f\n", out, bias, d[0], d[1], d[2]);*/
        /*if(fabs(out - tmp) < 1e-6) {*/
        /*printf("break\n");*/
        /*break;*/
        /*}*/
        /*out = tmp;*/
    }


    /*float theta = 0.05;*/
    /*for(int loop = 0; loop < 10; loop++) {*/
    /*[>softMax(d, 3);<]*/
    /*softMax_diff(d, 3);*/
    /*for(int i = 0; i < 3; i++) {*/
    /*printf("d:%0.3f\n", d[i]);*/
    /*}*/
    /*printf("-----------------------\n");*/
    /*[>float_t gard = 2 * pfc_layer->bias[0] -2;<]*/
    /*[>pfc_layer->bias[0] -= gard * theta;<]*/

    /*[>printf("loop:%d\tgard:%8.3f\tbias: %8.3f\n",loop,gard,pfc_layer->bias[0]);<]*/
    /*}*/
    /*float theta = 0.05;*/
    /*for(int loop = 0; loop < 50; loop++) {*/
    /*float_t gard = 2 * pfc_layer->bias[0] -2;*/
    /*pfc_layer->bias[0] -= gard * theta;*/

    /*printf("loop:%d\tgard:%8.3f\tbias: %8.3f\n",loop,gard,pfc_layer->bias[0]);*/
    /*}*/

    /*printf("-----------------------------\n");*/
//SGD moment
    /*pout_layer->output;*/
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

void destory_layer(union store_layer * player)
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
        if(player->pfc_layer->bias) {
            free(player->pfc_layer->bias);
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
        if(player->pout_layer->input) {
            free(player->pout_layer->input);
        }
        free(player->pout_layer);
    }
    break;

    default:
        break;
    }
}
