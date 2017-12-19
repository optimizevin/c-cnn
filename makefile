CC = gcc
CFLAGS += -g -std=c99  -Wall
#CLIBS += -lglib-2.0 -lpthread `pkg-config --libs opencv`   

tobj =  t.o  mnist.o
t: $(tobj)
	$(CC)  -o t $(tobj)

mnist.o : mnist.h mnist.c 
t.o : t.c 

.PHONY : clean 
clean : 
	-rm t $(tobj)

