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


/*1, 2, 3, 4,*/
/*5, 6, 7, 8,*/
/*9,10,11,12*/

/*0.1 0*/
/*0.1 0 */

/*0.1+0+0.5+0 = 0.6*/

int test()
{
    float_t sigf = 1.105905967;
    printf("sigmoid %.8f\n", sigmoid(sigf));

    float_t b[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    float_t f[] = {0.1, 0.f, 0.1, 0.f};

    /*ave_pool(b, 3, 4, 2, NULL);*/
    /*return 0;*/

    float_t*pOut = NULL;
    pOut = (float_t*)calloc(2 * 3, sizeof(float_t));

    conv2d_withonefilter(b, 3, 4, f, 2, 2, pOut);

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 3; j++) {
            float_t(*p)[][3] = (float_t(*)[][3])pOut;
            printf("%0.3f\t", (*p)[i][j]);
            /*printf("%0.3f",pOut[i*j+j]);*/
        }
        printf("\n");
    }
    return 0;


    /*float_t b[] = {2.f, 0.5, 1.f, 0.1, 1.f, 3.f};*/
    /*float_t blabel[] = {0.2, 0.3, 0.5, 0.1, 0.6, 0.3};*/
    /*float_t out[2] ;*/
    /*softMax_cross_entropy_with_logits(blabel,b,2,3,out);*/
    /*printf("%0.6f\t%0.6f  reduce:%0.6f\n",out[0],out[1], reduce_ment(out,2));*/

    /*return 0;*/


    /*for(;;) {*/
    /*uint32_t a, b;*/
    /*scanf("%d%d", &a, &b);*/
    /*uint32_t k[a][b];*/
    /*uint32_t  ka = CHECK_ROWS(k);*/
    /*uint32_t  kb = CHECK_COLS(k);*/
    /*printf("ka size: %d\nkb size:%d\n", ka, kb);*/
    /*}*/

    /*[>uint32_t (*p)[28];<]*/
    /*return 0;*/

}

uint32_t *pint_img  =  NULL;
uint32_t *pint_label  =  NULL;

void loadall()
{
    loadMnistImg(train_img_idx, &pint_img);
    loadMnistLabel(train_label_idx, &pint_label);
}

void initNet()
{
    union store_layer players[8];
    players[0].pinputlayer = create_inputlayer("input", (float_t*)pint_img, 28, 28, 1 , 0.5, 0.8);
    players[1].pconvlayer = create_convlayer("conv1", 6, 6, 8, 0.5, 0.8);

    conv2d_withlayer(players[0].pinputlayer->neu,28,28,players[1].pconvlayer);

    /*players[0] = makelayer("input", 28, 28, 1, 0.5, 0.8, LAY_INPUT);*/
    /*players[1] = makelayer("conv1", 8, 6, 6, 0.5, 0.8, LAY_CONV);*/
    /*players[2] = makelayer("s2",2,2,1,0.5,0.8,LAY_POOL);*/
    /*subsampling_fun();*/
    /*players[3] = makelayer("conv3",5,5,16,0.5,0.8,LAY_CONV);*/
    /*players[4] = makelayer("s4",2,2,1,0.5,0.8,LAY_SUBSAM);*/
    /*players[5] = makelayer("conv5",5,5,120,0.5,0.8,LAY_CONV);*/
    /*players[6] = makelayer("FullyConnect",28,28,1,0.5,0.8,LAY_FULLYCONNECT);*/
    /*players[7] = makelayer("Output",10,1,1,0.5,0.8,LAY_OUT);*/


    /*const uint32_t size = 28*28;*/
    /*for(uint32_t i=0;i<60000;i++){*/
    /*const uint32_t step = i*size;*/
    /*memcpy(players[0]->neu,(float_t*)pint_img+step,size);*/
    /*}*/
    /*core_forward(players, 2, 0.01);*/

    destory_convlayer(players[1].pconvlayer);
    destory_inputlayer(players[0].pinputlayer);
    /*SAFEFREE(players[1]->pconv_filter);*/
}


int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    printf("test\n");
    /*test();*/

    loadall();
    /*pint_img*/
    initNet();

    free(pint_img);
    free(pint_label);

    return 0;
}
