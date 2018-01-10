/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : cnn.h
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*   GitHub      : https://github.com/optimizevin
*/

#include <stdio.h>
#include <stdint.h>
#include "nncomm.h"

struct conv_filter_head {
    uint32_t  in_height;
    uint32_t  in_width;
    uint32_t  filter_batch;
    float     filter_core[0];
};

struct conv_filter_head *create_convcore(const uint32_t batch, const uint32_t height,
        const uint32_t width,const float mu,const float stddev);

inline struct data_batch *conv2d(const struct data_batch * pdatabatch, 
        struct conv_filter_head * filter, const int strides, const int padding);
//max_pool_2x2;



