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

/*1, 2, 3, 4,*/
/*5, 6, 7, 8,*/
/*9,10,11,12*/

/*0.1 0*/
/*0.1 0 */

/*0.1+0+0.5+0 = 0.6*/


int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    printf("test\n");

    float_t b[]={1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f,9.f,10.f,11.f,12.f};
    float_t f[] = {0.1,0.f,0.1,0.f};

    float_t*pOut = NULL;
    pOut = (float_t*)calloc(2*3,sizeof(float_t));

    conv2d_withonefilter(b, 3, 4, f, 2,2, pOut);

    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            float_t(*p)[][3] = (float_t(*)[][3])pOut;
            printf("%0.3f\t",(*p)[i][j]);
            /*printf("%0.3f",pOut[i*j+j]);*/
        }
        printf("\n");
    }
    return 0;
    /*void conv2d(const float_t *pData, uint32_t data_height, uint32_t data_width, float_t *filter, uint32_t fl_height, uint32_t fl_width, float_t *pOut)*/

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


    uint32_t ret = 0;
    uint32_t *pint_img  =  NULL;
    ret = loadMnistImg(train_img_idx, &pint_img);
    printf("loadMnistImg ret = %i32u\n", ret);


    uint32_t (*pi)[28] = (void*)pint_img;

    for(int i = 0; i < 60000; i++) {
        for(int j = 0; j < 28; j++) {
            for(int k = 0; k < 28; k++) {
                /*int tmp = pi[j][k];*/
            }
        }
        pi++;
    }

    /*pint_img*/


    uint32_t *pint_label  =  NULL;
    ret = loadMnistLabel(train_label_idx, &pint_label);
    printf("loadMnistLabel ret = %i32u\n", ret);
    free(pint_img);
    free(pint_label);



    return 0;
}
