/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : comm.h
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/

struct binaryImg{
	unsigned int hiegh;
	unsigned int width;
	unsigned int bImg[0];
};


binaryImg *loadImg(const char* filename);
