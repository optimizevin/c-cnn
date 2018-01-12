/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : c-cnn.c
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

#include "stdlib.h"
#include "stdio.h"

#include "nncomm.h"
#include "time.h"


void init()
{
    srand((unsigned)time(NULL));
}

int main(int argc,char**argv)
{
    printf("cnn...\n");
    init();

    long long ret = 0;
    int *pint_img  =  NULL;
    ret = loadMnistImg(train_img_idx,&pint_img);
    printf("loadMnistImg ret = %lld\n",ret);
    

    int (*pi)[28] = (void*)pint_img;
    /*struct mnist_img* si = (struct mnist_img*)pint_img;*/
    for(int i=0;i<60000;i++){
       for(int j=0;j<28;j++){
           for(int k=0;k<28;k++){
               int tmp = pi[j][k];
           }
       } 
       pi++;
    }
   
    /*pint_img*/


    int *pint_label  =  NULL;
    ret = loadMnistLabel(train_label_idx,&pint_label);

    printf("loadMnistLabel ret = %lld\n",ret);
    free(pint_img);
    free(pint_label);
    return 0;
}
