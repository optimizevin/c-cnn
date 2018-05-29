/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : t.c
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*  GitHub      : https://github.com/optimizevin
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

#include "nncomm.h"
#include "cnn.h"
#include "mnist.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>
#include <assert.h>
#include "float.h"


/*1, 2, 3, 4,*/
/*5, 6, 7, 8,*/
/*9,10,11,12*/

/*0.1 0*/
/*0.1 0 */

/*0.1+0+0.5+0 = 0.6*/

int test()
{
    /*for(int i=0;i<100;i++){*/
    /*float_t sigf = generateGaussianNoise(0.5,0.8f);*/
    /*[>if(sigf>1.f || sigf < -1.f)<]*/
    /*printf("%5.3f\n",sigf);*/
    /*}*/

    /*return 0;*/


    float_t sigf = 1.105905967;
    printf("sigmoid %.8f\n", sigmoid(sigf));

    /*float_t b[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};*/
    /*float_t f[] = {0.1, 0.f, 0.1, 0.f};*/


    /*ave_pool(b, 3, 4, 2, NULL);*/
    /*return 0;*/

    float_t*pOut = NULL;
    pOut = (float_t*)calloc(2 * 3, sizeof(float_t));

    /*conv2d_withonefilter(b, 3, 4, f, 2, 2, 0.5f, pOut);*/

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 3; j++) {
            float_t(*p)[][3] = (float_t(*)[][3])pOut;
            printf("%0.3f\t", (*p)[i][j]);
            /*printf("%0.3f",pOut[i*j+j]);*/
        }
        printf("\n");
    }
    return 0;

}

void  test_softmax()
{
    /*float_t b[] = {2.f, 0.5, 1.f, 0.1, 1.f, 3.f};*/
    /*float_t blabel[] = {0.2, 0.3, 0.5, 0.1, 0.6, 0.3};*/
    float_t b[] = {3442415.750, 257349.219, 3287622.5, 0.5f, 805999.875, 3328236.750, 242296.984, 2133911.750, 1361594.0, 994116.125};


    /*float_t c[] = {0.2,0.3,0.5};*/
    float_t c[] = {1.0,2.0,3.0};
    /*float_t bl[] = {0.f, 1.f, 0.f};*/
    /*softMax(c,3);*/
    /*for(int i=0;i<3;i++)*/
    /*printf("c:%8.4f\n",c[i]);*/
    float_t o = 0.f;
    /*inline void softMax_cross_entropy_with_logits(const float_t *labels, float_t *logits, const uint32_t blen,float_t *pOut)*/
    softMax_cross_entropy_with_logits(2, c, 3, &o);

    printf("reto:%8.4f\n", o);
    return;

    /*float_t blabel[] = {0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f};*/
    //blabel[1.f,1.f]
    float_t out[2] = {0} ;
    /*MinMax(b, 1, 10);*/
    for(int i = 0; i < 10; i++) {
        printf("minmax:%4.8f\n", b[i]);
    }
    /*smf(b,blabel,10);*/
    softMax_cross_entropy_with_logits(1, b, 10, out);
    printf("2X3   %0.6f\t%0.6f  reduce_mean:%0.6f\n", out[0], out[1], reduce_mean(out, 1));


    /*[>float_t pb[] = {3.0, 1.0, 0.2, 5.0, 7.12, 10.0};<]*/
    /*[>float_t pb[] = {3442415.750, 257349.219, 3287622.5, 0.5f, 805999.875, 3328236.750, 242296.984, 2133911.750, 1361594.0, 994116.125};<]*/
    /*float_t pb[] = {1.f,2.f,3.f};*/
    /*for(int i = 0; i < sizeof(pb)/sizeof(pb[0]); i++) {*/
    /*printf("pb:%4.8f\n", pb[i]);*/
    /*}*/
    /*[>MinMax(pb,1,sizeof(pb)/sizeof(pb[0]));<]*/
    /*softMax_cross(pb, 1,sizeof(pb)/sizeof(pb[0]));*/
    /*printf("softMaxloss:\n");*/
    /*float sum = 0.f;*/
    /*for(int i = 0; i < sizeof(pb)/sizeof(pb[0]); i++) {*/
    /*printf("%5.6f\n", pb[i]);*/
    /*sum +=pb[i];*/
    /*}*/
    /*printf("sum:%8.3f",sum);*/
    /*printf("\n");*/

    return ;
}

void test_maxpool()
{
    float_t b[] = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        9.f, 10.f, 11.f, 12.f,
        13.f, 14.f, 15.f, 16.f
    };
    float_t out[64] = {0};
    max_pool(b, 4, 4, 2, 2, 2, out);
}

/*static list_declare(elment_list);*/

void test_slist()
{


}

float_t *pint_img  =  NULL;
uint32_t *pint_label  =  NULL;

void loadall()
{
    loadMnistImg(train_img_idx, &pint_img);
    loadMnistLabel(train_label_idx, &pint_label);
}

#define  PREV_LOAD    20
#define  LAYER_BATCH    10
#define  FILTER_CORE_BATCH  8
#define  MAX_POOL_STRIDE  2
#define  BIAS  0.1f
#define  CLASS_LOOP 10

void initNet()
{
    union store_layer P[LAYER_BATCH];
    /*for(int i=0;i<20;i++){*/
    /*printf("label:%d\n",pint_label[i]);*/
    /*}*/
    P[0].pinput_layer = create_inputlayer("input", 28, 28);
    /*logpr(P[0].pinput_layer->pdata,*/
    /*P[0].pinput_layer->in_rows,*/
    /*P[0].pinput_layer->in_cols, 19);*/

    P[1].pconv_layer = create_convlayer("conv1", 7, 7, FILTER_CORE_BATCH, BIAS, 0.8);
    /*28-7+1 = 22*/
    P[2].ppool_layer = create_poollayer("pool", 5, 5);
    P[3].pconv_layer = create_convlayer("conv2", 3, 3, FILTER_CORE_BATCH, BIAS, 0.8);
    P[4].ppool_layer = create_poollayer("pool2", 2, 2);
    P[5].pfc_layer = create_fully_connected_layer("fully connection 1/2", 10, 4 * 4 * 64,0.5); 
    P[6].poutput_layer = create_output_layer("output layer", CLASS_LOOP);
    /*P[6].pdrop_layer = create_dropout_layer("dropout layer 2", 1, 1, 64);*/
    /*P[7].pfc_layer = create_fully_connected_layer("fully connection 2/2", 64, CLASS_LOOP, 0.5f);*/
    /*P[8].poutput_layer = create_output_layer("output layer", CLASS_LOOP);*/

    const uint32_t imgsize = 28 * 28;

    for(uint32_t  batch = 0; batch < 1; batch++) {
        load_inputlayer(P[0].pinput_layer, pint_img + batch * imgsize, pint_label[batch]);
        /*MinMax((float_t*)(P[0].pinput_layer->pdata), 28 ,28);*/
        /*logpr((float_t*)(P[0].pinput_layer->pdata), 28 ,28,0);*/
        conv2d_withlayer(P[0].pinput_layer->pdata, 28, 28, 1, P[1].pconv_layer);
        /*printf("conv1 rows:%4d\tconv cols:%4d\tconv batch:%4d\n",*/
        /*P[1].pconv_layer->out_rows,*/
        /*P[1].pconv_layer->out_cols,*/
        /*P[1].pconv_layer->out_batch);*/


        pool_withlayer(P[1].pconv_layer->conv_out,
                       P[1].pconv_layer->out_rows,
                       P[1].pconv_layer->out_cols,
                       P[1].pconv_layer->out_batch,
                       P[2].ppool_layer, MAX_POOL_STRIDE );

        /*printf("pool1 rows:%4d\tpool cols:%4d\tpool batch:%4d\n",*/
        /*P[2].ppool_layer->out_rows,*/
        /*P[2].ppool_layer->out_cols,*/
        /*P[2].ppool_layer->out_batch);*/

        /*logpr(P[2].ppool_layer->poolout,*/
        /*P[2].ppool_layer->out_rows,*/
        /*P[2].ppool_layer->out_cols, 31);*/

        conv2d_withlayer(P[2].ppool_layer->pool_out, P[2].ppool_layer->out_rows,
                         P[2].ppool_layer->out_cols, P[2].ppool_layer->out_batch, P[3].pconv_layer);
        /*printf("conv2 rows:%4d\tconv cols:%4d\tconv batch:%4d\n",*/
        /*P[3].pconv_layer->out_rows,*/
        /*P[3].pconv_layer->out_cols,*/
        /*P[3].pconv_layer->out_batch);*/
        /*logpr(P[3].pconvlayer->pout,*/
        /*P[3].pconvlayer->out_rows,*/
        /*P[3].pconvlayer->out_cols, 6);*/

        pool_withlayer(P[3].pconv_layer->conv_out,
                       P[3].pconv_layer->out_rows,
                       P[3].pconv_layer->out_cols,
                       P[3].pconv_layer->out_batch,
                       P[4].ppool_layer, MAX_POOL_STRIDE );


        /*printf("pool2 rows:%4d\tpool cols:%4d\tpool batch:%4d\n",*/
               /*P[4].ppool_layer->out_rows,*/
               /*P[4].ppool_layer->out_cols,*/
               /*P[4].ppool_layer->out_batch);*/

        /*logpr(P[4].ppool_layer->pool_out,*/
              /*P[4].ppool_layer->out_rows,*/
              /*P[4].ppool_layer->out_cols, 0);*/


        fully_connected_fclayer( P[4].ppool_layer->pool_out,
                                 P[4].ppool_layer->out_rows,
                                 P[4].ppool_layer->out_cols,
                                 P[4].ppool_layer->out_batch,
                                 P[5].pfc_layer);

        /*printf("%s neunum:%4d\n", P[5].pfc_layer->base.layerName, P[5].pfc_layer->neunum);*/
        /*logpr(P[5].pfc_layer->neu, 1, P[5].pfc_layer->neunum, 0);*/

        /*dropout_layer( P[5].pfc_layer->neu,*/
        /*1, 1,*/
        /*P[5].pfc_layer->neunum,*/
        /*P[6].pdrop_layer);*/
        /*printf("dropout rows:%4d  dropout cols:%4d\tdroplayer batch:%4d\n",*/
        /*P[6].pdrop_layer->out_rows,*/
        /*P[6].pdrop_layer->out_cols,*/
        /*P[6].pdrop_layer->drop_batch);*/

        /*logpr(P[6].pdrop_layer->drop_out,*/
        /*1,*/
        /*64, 0);*/

        /*fully_connected_fclayer( P[6].pdrop_layer->drop_out,*/
        /*1, 1, P[5].pfc_layer->neunum,*/
        /*P[7].pfc_layer);*/

        /*printf("%s neunum:%4d\n", P[7].pfc_layer->base.layerName,*/
        /*P[7].pfc_layer->neunum);*/
        /*logpr(P[7].pfc_layer->neu,*/
        /*1, P[7].pfc_layer->neunum, 0);*/

        /*P[8].poutput_layer->output[loop] = P[7].pfc_layer->neu[0];*/

        /*output_epoch(P[7].pfc_layer, P[8].poutput_layer, P[0].pinput_layer->label,10);*/
        /*output_epoch(P[5].pfc_layer, P[6].poutput_layer, P[0].pinput_layer->label, 10);*/
        output_epoch_bias(P[5].pfc_layer, P[6].poutput_layer, P[0].pinput_layer->label, 10,
         P[4].ppool_layer->pool_out,
                                 P[4].ppool_layer->out_rows,
                                 P[4].ppool_layer->out_cols,
                                 P[4].ppool_layer->out_batch);
        /*output_epoch(P[5].pfc_layer, P[6].poutput_layer, P[0].pinput_layer->label, 10,*/
         /*P[4].ppool_layer->pool_out,*/
                                 /*P[4].ppool_layer->out_rows,*/
                                 /*P[4].ppool_layer->out_cols,*/
                                 /*P[4].ppool_layer->out_batch);*/

        /*for(uint32_t loop = 0; loop < 10; loop++) {*/
        /*printf("input:%4.3f\n", P[8].poutput_layer->input[loop]);*/
        /*}*/
        /*printf("output:%8.3f\t feture:%d\n", P[8].poutput_layer->output, P[0].pinput_layer->label);*/
    }

    /*destory_layer(&P[8]);*/
    /*destory_layer(&P[7]);*/
    destory_layer(&P[6]);
    destory_layer(&P[5]);
    destory_layer(&P[4]);
    destory_layer(&P[3]);
    destory_layer(&P[2]);
    destory_layer(&P[1]);
    destory_layer(&P[0]);

}



int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    printf("test\n");
    /*[>[>[>test();<]<]<]*/
    /*test_softmax();*/
    /*return 0;*/

    loadall();
    /*pint_img*/
    initNet();

    free(pint_img);
    free(pint_label);

    return 0;
}
