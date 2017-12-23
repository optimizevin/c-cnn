/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : cnn.h
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/

#include <stdio.h>
#include <stdint.h>

struct conv_filter_head {
    uint32_t  in_height;
    uint32_t  in_width;
    uint32_t  filter_batch;
    float     filter_core[0];
};

inline  float* randf(const uint32_t nsize, const float stddev);
struct conv_filter_head *create_convcore(const uint32_t batch, const uint32_t height,
        const uint32_t width,const float mu,const float stddev);
//max_pool_2x2



