/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : t.c
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/

#include "nncomm.h"
#include "cnn.h"
#include "mnist.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>


/*typedef  struct  pimg{*/
/*int h;*/
/*int w;*/
/*union{*/
/*int (*p)[28];*/
/*int  bin[0];*/
/*}*/
/*}ppimg;*/


int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    printf("test\n");

	/* 50  60 */
	/* 80  90 */

	float b[]={10.f,20.f,30.f,40.f,50.f,60.f,70.f,80.f,90.f,100.f,110.f,120.f};
	float out[] = {0.f,0.f,0.f,0.f,0.f,0.f};
	max_pool(b,3,4,2,out);
    for(int i=0;i<sizeof(out)/sizeof(out[0]);i++)
        printf("%f\n",out[i]);

    
    return 0;


    for(;;) {
        uint32_t a, b;
        scanf("%d%d", &a, &b);
        uint32_t k[a][b];
        uint32_t  ka = CHECK_ROWS(k);
        uint32_t  kb = CHECK_COLS(k);
        printf("ka size: %d\nkb size:%d\n", ka, kb);
    }

    /*uint32_t (*p)[28];*/
    return 0;


    uint32_t ret = 0;
    uint32_t *pint_img  =  NULL;
    ret = loadMnistImg(train_img_idx, &pint_img);
    printf("loadMnistImg ret = %i32u\n", ret);


    uint32_t (*pi)[28] = (void*)pint_img;

    for(int i = 0; i < 60000; i++) {
        for(int j = 0; j < 28; j++) {
            for(int k = 0; k < 28; k++) {
                /*int tmp = pi[j][k];*/
            }
        }
        pi++;
    }

    /*pint_img*/


    uint32_t *pint_label  =  NULL;
    ret = loadMnistLabel(train_label_idx, &pint_label);
    printf("loadMnistLabel ret = %i32u\n", ret);
    free(pint_img);
    free(pint_label);



    return 0;
}
