/*  Copyright (C), 2017-1, Tech. Co., Ltd.*/
/*  File name: c-cnn.c
/*  Author: vincent  */
/*  Version: 0.9  */
/*  Date: 2017.6
/*  Description: cnn */

#include "stdlib.h"
#include "stdio.h"

#include "comm.h"

int main(int argc,char**argv)
{
    printf("cnn...\n");

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
