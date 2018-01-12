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


/*typedef  struct  pimg{*/
/*int h;*/
/*int w;*/
/*union{*/
/*int (*p)[28];*/
/*int  bin[0];*/
/*}*/
/*}ppimg;*/


int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    printf("test\n");

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
