/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   :  mnist.h
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*   GitHub      : https://github.com/optimizevin
*/

//0000     32 bit integer  0x00000803(2051) magic number
//0004     32 bit integer  60000            number of images
//0008     32 bit integer  28               number of rows
//0012     32 bit integer  28               number of columns
//
#include "stdio.h"
#include "stdlib.h"
#include <stdint.h>
#include <math.h>

#define  t10k_img_idx       "./mnistdb/t10k-images.idx3-ubyte"
#define  t10k_label_idx     "./mnistdb/t10k-labels.idx1-ubyte"
#define  train_img_idx      "./mnistdb/train-images.idx3-ubyte"
#define  train_label_idx    "./mnistdb/train-labels.idx1-ubyte"



struct mnist_img{
    uint32_t rows[28];
    uint32_t cols[28];
};

struct mnist_pixel_file {
    uint32_t  msb;                         // 32 bit integer  0x00000803(2051) magic number
    uint32_t  num;                         // 32 bit integer  60000            number of images
    uint32_t  num_rows;                    // 32 bit integer  28               number of rows
    uint32_t  num_cols;                    // 32 bit integer  28               number of columns
    unsigned char  pixel[0];
};


struct mnist_label_file {
    uint32_t msb;                          // 32 bit integer  0x00000801(2049) magic number (MSB first)
    uint32_t num;                          // 32 bit integer  60000            number of items
    unsigned char pixel[0];          // unsigned byte   ??               label
};

uint32_t loadMnistImg(const char* filename, float_t **p);
uint32_t loadMnistLabel(const char *filename, uint32_t **p);
