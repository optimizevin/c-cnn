/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : t.c
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/


#include "stdlib.h"
#include "stdio.h"
#include "mnist.h"
#include "memory.h"


int main(int argc,char **argv)
{
    printf("test\n");
    long long ret = 0;
    int *pint_img  =  NULL;
    ret = loadMnistImg(train_img_idx,&pint_img);
    printf("loadMnistImg ret = %lld\n",ret);
    int *pint_label  =  NULL;
    ret = loadMnistLabel(train_label_idx,&pint_label);
    printf("loadMnistLabel ret = %lld\n",ret);
    free(pint_img);
    free(pint_label);
    return 0;
}
