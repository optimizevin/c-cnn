/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   :  mnist.c
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/

#include "mnist.h"
#include <sys/stat.h>  
#include <memory.h>

static inline int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


struct binaryImg *loadImg(const char* filename)
{
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp){
        printf("load file:%s faile\n",filename);
        return 0;
    }
    struct stat st; 
    stat(filename, &st); 
    int fileSize =  st.st_size; 
    struct binaryImg* pbimg = (struct binaryImg*)malloc(sizeof(struct binaryImg) + fileSize);
    fclose(fp);
    return pbimg;
}

long long loadMnistImg(const char* filename,int **p)
{
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp){
        printf("load file:%s faile\n",filename);
        return 0;
    }
    struct stat st;
    stat(filename, &st);
    int fileSize =  st.st_size;
    struct mnist_pixel_file *pmnistpf = (struct mnist_pixel_file *)malloc(fileSize);
    memset(pmnistpf,0x0,fileSize);
    fread(pmnistpf,fileSize,1,fp);
    fclose(fp);

    pmnistpf->num =  reverseInt(pmnistpf->num);
    pmnistpf->num_rows =  reverseInt(pmnistpf->num_rows);
    pmnistpf->num_cols =  reverseInt(pmnistpf->num_cols);

    long long ret = pmnistpf->num;
    long long count =  pmnistpf->num * pmnistpf->num_rows * pmnistpf->num_cols;

    int *pInt = (int*)calloc(count,sizeof(int));
    for(int i=0;i<count;i++){
        pInt[i] =  reverseInt( pmnistpf->pixel[i]);
    }
    free(pmnistpf);
    *p = pInt;
    return ret;
}


long long loadMnistLabel(const char *filename,int **p)
{
    FILE *fp = fopen(filename, "rb");
    if (NULL == fp){
        printf("load file:%s faile\n",filename);
        return 0;
    }
    struct stat st;
    stat(filename, &st);
    int fileSize =  st.st_size;
    struct mnist_label_file *pmlf = (struct mnist_label_file *)malloc(fileSize);
    memset(pmlf,0x0,fileSize);
    fread(pmlf,fileSize,1,fp);
    fclose(fp);

    pmlf->num =  reverseInt(pmlf->num);
    long long ret = pmlf->num;
    int *pInt = (int*)calloc(ret,sizeof(int));
    memset(pInt,0x0,(ret*sizeof(int)));

    for(int i=0;i<ret;i++){
        pInt[i] =  (int)pmlf->pixel[i];
        /*printf("%d:  %d\n",i,pInt[i]);*/
    }
    free(pmlf);
    *p = pInt;
    return ret;
}


/*ImgArr read_Img(const char* filename) // 读入图像*/
/*{*/
    /*FILE  *fp = NULL;*/
    /*fp = fopen(filename, "rb");*/
    /*if(fp == NULL)*/
        /*printf("open file failed\n");*/
    /*assert(fp);*/

    /*int magic_number = 0;*/
    /*int number_of_images = 0;*/
    /*int n_rows = 0;*/
    /*int n_cols = 0;*/
    /*//从文件中读取sizeof(magic_number) 个字符到 &magic_number*/
    /*fread((char*)&magic_number, sizeof(magic_number), 1, fp);*/
    /*magic_number = ReverseInt(magic_number);*/
    /*//获取训练或测试image的个数number_of_images*/
    /*fread((char*)&number_of_images, sizeof(number_of_images), 1, fp);*/
    /*number_of_images = ReverseInt(number_of_images);*/
    /*//获取训练或测试图像的高度Heigh*/
    /*fread((char*)&n_rows, sizeof(n_rows), 1, fp);*/
    /*n_rows = ReverseInt(n_rows);*/
    /*//获取训练或测试图像的宽度Width*/
    /*fread((char*)&n_cols, sizeof(n_cols), 1, fp);*/
    /*n_cols = ReverseInt(n_cols);*/
    /*//获取第i幅图像，保存到vec中*/
    /*int i, r, c;*/

    /*// 图像数组的初始化*/
    /*ImgArr imgarr = (ImgArr)malloc(sizeof(MinstImgArr));*/
    /*imgarr->ImgNum = number_of_images;*/
    /*imgarr->ImgPtr = (MinstImg*)malloc(number_of_images * sizeof(MinstImg));*/

    /*for(i = 0; i < number_of_images; ++i) {*/
        /*imgarr->ImgPtr[i].r = n_rows;*/
        /*imgarr->ImgPtr[i].c = n_cols;*/
        /*imgarr->ImgPtr[i].ImgData = (float**)malloc(n_rows * sizeof(float*));*/
        /*for(r = 0; r < n_rows; ++r) {*/
            /*imgarr->ImgPtr[i].ImgData[r] = (float*)malloc(n_cols * sizeof(float));*/
            /*for(c = 0; c < n_cols; ++c) {*/
                /*// 因为神经网络用float型计算更为精确，这里我们将图像像素转为浮点型*/
                /*unsigned char temp = 0;*/
                /*fread((char*) &temp, sizeof(temp), 1, fp);*/
                /*imgarr->ImgPtr[i].ImgData[r][c] = (float)temp / 255.0;*/
            /*}*/
        /*}*/
    /*}*/

    /*fclose(fp);*/
    /*return imgarr;*/
/*}*/

/*（3）读入图像数据标号*/
/*[cpp] view plain copy*/
/*LabelArr read_Lable(const char* filename)// 读入图像*/
/*{*/
    /*FILE  *fp = NULL;*/
    /*fp = fopen(filename, "rb");*/
    /*if(fp == NULL)*/
        /*printf("open file failed\n");*/
    /*assert(fp);*/

    /*int magic_number = 0;*/
    /*int number_of_labels = 0;*/
    /*int label_long = 10;*/

    /*//从文件中读取sizeof(magic_number) 个字符到 &magic_number*/
    /*fread((char*)&magic_number, sizeof(magic_number), 1, fp);*/
    /*magic_number = ReverseInt(magic_number);*/
    /*//获取训练或测试image的个数number_of_images*/
    /*fread((char*)&number_of_labels, sizeof(number_of_labels), 1, fp);*/
    /*number_of_labels = ReverseInt(number_of_labels);*/

    /*int i, l;*/

    /*// 图像标记数组的初始化*/
    /*LabelArr labarr = (LabelArr)malloc(sizeof(MinstLabelArr));*/
    /*labarr->LabelNum = number_of_labels;*/
    /*labarr->LabelPtr = (MinstLabel*)malloc(number_of_labels * sizeof(MinstLabel));*/

    /*for(i = 0; i < number_of_labels; ++i) {*/
        /*// 数据库内的图像标记是一位，这里将图像标记变成10位，10位中只有唯一一位为1，为1位即是图像标记*/
        /*labarr->LabelPtr[i].l = 10;*/
        /*labarr->LabelPtr[i].LabelData = (float*)calloc(label_long, sizeof(float));*/
        /*unsigned char temp = 0;*/
        /*fread((char*) &temp, sizeof(temp), 1, fp);*/
        /*labarr->LabelPtr[i].LabelData[(int)temp] = 1.0;*/
    /*}*/

    /*fclose(fp);*/
    /*return labarr;*/
/*}*/
