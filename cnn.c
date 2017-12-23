/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : cnn.c
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/
#include "cnn.h"
#include "comm.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

struct conv_filter_head *create_convcore(const uint32_t batch, const uint32_t height,
        const uint32_t width,const float mu,const float stddev)
{
    const uint32_t size = height * width * batch + sizeof(struct conv_filter_head);
    struct conv_filter_head *pfh = (struct conv_filter_head*)malloc(size);
    memset(pfh, 0x0,size);
    for(uint32_t  i=0;i<size;i++){
        pfh->filter_core[i] =  generateGaussianNoise(mu, stddev);
    }
    /*float (*fit)[width] = (float(*)[width])pfh->filter_core;*/
    return pfh;

}

//第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
//具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
//注意这是一个4维的Tensor，要求类型为float32和float64其中之一
//第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
//第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
//第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）

inline void conv2d(const struct data_batch * pbatch, void * filter, const int strides, const int padding)
{
}

