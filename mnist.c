/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   :  mnist.c
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*   GitHub      : https://github.com/optimizevin
*/

#include "mnist.h"
#include <sys/stat.h>
#include <memory.h>
#include "assert.h"

static inline int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


/*uint32_t erloadMnistImg(const char* filename, float_t **p)*/
/*{*/
    /*FILE *fp = fopen(filename, "rb");*/
    /*if (NULL == fp) {*/
        /*printf("load file:%s faile\n", filename);*/
        /*return 0;*/
    /*}*/
    /*struct stat st;*/
    /*stat(filename, &st);*/
    /*uint32_t fileSize =  st.st_size;*/
    /*assert(fileSize > 0);*/
    /*struct mnist_pixel_file *pmnistpf = (struct mnist_pixel_file *)calloc(fileSize, 1);*/
    /*fread(pmnistpf, sizeof(struct mnist_pixel_file) , 1, fp);*/

    /*pmnistpf->num =  reverseInt(pmnistpf->num);*/
    /*pmnistpf->num_rows =  reverseInt(pmnistpf->num_rows);*/
    /*printf("rows:%d\n", pmnistpf->num_rows);*/
    /*pmnistpf->num_cols =  reverseInt(pmnistpf->num_cols);*/
    /*printf("cols:%d\n", pmnistpf->num_cols);*/
    /*uint32_t batch = pmnistpf->num;*/
    /*uint32_t count =  pmnistpf->num * pmnistpf->num_rows * pmnistpf->num_cols;*/

    /*float_t *pfloat = (float_t*)calloc(count, sizeof(float_t));*/

    /*float_t(*ptmp)[][28] = (float_t(*)[][28])pfloat;*/
    /*for(int num = 0; num < batch; num++) {*/
        /*for(int rows = 0; rows < pmnistpf->num_rows; rows++) {*/
            /*for(int cols = 0; cols < pmnistpf->num_cols; cols++) {*/
                /*unsigned char tmp = '0';*/
                /*fread(&tmp, sizeof(unsigned char), 1, fp);*/
                /*(*ptmp)[rows][cols] = (float_t)tmp;*/
                /*[>printf("%05.1f ", (*ptmp)[rows][cols]);<]*/
            /*}*/
            /*[>printf("\n");<]*/
        /*}*/
    /*}*/

    /*fclose(fp);*/
    /*free(pmnistpf);*/
    /**p = pfloat;*/
    /*return batch;*/
/*}*/


uint32_t loadMnistImg(const char* filename, float_t **p)
{
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("load file:%s faile\n", filename);
        return 0;
    }
    struct stat st;
    stat(filename, &st);
    uint32_t fileSize =  st.st_size;
    struct mnist_pixel_file *pmnistpf = (struct mnist_pixel_file*)calloc(fileSize,1);
    fread(pmnistpf, fileSize, 1, fp);
    fclose(fp);

    pmnistpf->num =  reverseInt(pmnistpf->num);
    pmnistpf->num_rows =  reverseInt(pmnistpf->num_rows);
    pmnistpf->num_cols =  reverseInt(pmnistpf->num_cols);

    uint32_t count =  pmnistpf->num * pmnistpf->num_rows * pmnistpf->num_cols;

    float_t *pfloat = (float_t*)calloc(count, sizeof(float_t));
    for(int i = 0; i < count; i++) {
        pfloat[i] = (float_t)pmnistpf->pixel[i];
    }
    uint32_t ret =  pmnistpf->num;
    free(pmnistpf);
    *p = pfloat;
    return ret;
}


uint32_t loadMnistLabel(const char *filename, uint32_t **p)
{
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("load file:%s faile\n", filename);
        return 0;
    }
    struct stat st;
    stat(filename, &st);
    uint32_t fileSize =  st.st_size;
    assert(fileSize > 0);
    struct mnist_label_file *pmlf = (struct mnist_label_file *)malloc(fileSize);
    memset(pmlf, 0x0, fileSize);
    fread(pmlf, fileSize, 1, fp);
    fclose(fp);

    pmlf->num =  reverseInt(pmlf->num);
    uint32_t batch = pmlf->num;
    uint32_t *pInt = (uint32_t*)calloc(batch, sizeof(uint32_t));

    for(uint32_t i = 0; i < batch; i++) {
        pInt[i] =  (uint32_t)pmlf->pixel[i];
    }
    free(pmlf);
    *p = pInt;
    return batch;
}
