/*
*   Copyright (C), 2017-1, Tech. Co., Ltd.
*   File name   : cnn.h
*   Author      : vincent
*   Version     : 0.9
*   Date        : 2017.6
*   Description : cnn
*/

#include <math.h>
#include <stdio.h>

#define sigmoid_exp fastexp

inline double fastexp(double x) {
	/* e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) + (x^4 / 4!) */

	double sum = 1 + x;
	double n = x;
	double d = 1;
	double i;

	for(i = 2; i < 100; i++) {
		n *= x;
		d *= i;
		sum += n / d;
	}

	return sum;
}

inline double sigmoid(double x) {
	return 1.00 / (1 + sigmoid_exp(0 - x));
}

#define Relu (x) x>0?x:0
//noisyRelu
//leakyRelu
