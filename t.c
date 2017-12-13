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


int main(int argc,char **argv)
{
    printf("test\n");
    struct mnist_pixel_pack  mpp;
    printf("add: %ld\n",&mpp);
    loadMnistImg(train_img_idx,&mpp);
    printf("add: %ld\n",&mpp);
    return 0;
}
