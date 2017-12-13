/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   :  mnist.h
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/

//0000     32 bit integer  0x00000803(2051) magic number 
//0004     32 bit integer  60000            number of images 
//0008     32 bit integer  28               number of rows 
//0012     32 bit integer  28               number of columns 
//
#include "stdio.h"
#include "stdlib.h"

#define  t10k_img_idx       "./mnistdb/t10k-images.idx3-ubyte"
#define  t10k_label_idx     "./mnistdb/t10k-labels.idx1-ubyte"
#define  train_img_idx      "./mnistdb/train-images.idx3-ubyte"
#define  train_label_idx    "./mnistdb/train-labels.idx1-ubyte"

struct binaryImg{
	unsigned int hight;
	unsigned int width;
	unsigned int bImg[0];
};

struct binaryImg_float{
	unsigned int hight;
	unsigned int width;
	float  bImg[0];
};

struct mnistImg{
	int hight;
	int width;
	float  bImg[0];
};

struct mnist_pixel_pack{
    unsigned long  long msb;          // 32 bit integer  0x00000803(2051) magic number 
    unsigned long  long num;          // 32 bit integer  60000            number of images 
    unsigned long  long num_rows;     // 32 bit integer  28               number of rows 
    unsigned long  long num_cols;     // 32 bit integer  28               number of columns 
    unsigned char *pixel[0];          // pixel
};

struct mnist_label_pack{
    unsigned long  long msb;          // 32 bit integer  0x00000801(2049) magic number (MSB first) 
    unsigned long  long num;          // 32 bit integer  60000            number of items 
    unsigned char *pixel[0];          // unsigned byte   ??               label 
};

int loadImg(const char* filename,struct binaryImg *loadImg);
int loadMnistImg(const char* filename,struct mnist_pixel_pack *mpp);
