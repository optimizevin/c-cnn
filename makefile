CC = gcc
CFLAGS += -g -ansi -std=c99 -Wall -fgnu89-inline
CLIBS += -lm
#CLIBS += -lglib-2.0 -lpthread `pkg-config --libs opencv`   

tobj =  t.o  mnist.o cnn.o  comm.o
t: $(tobj)
	$(CC) -o t $(tobj) $(CLIBS)

comm.o: comm.h comm.c
mnist.o : mnist.h mnist.c 
t.o : t.c 
cnn.o : cnn.h cnn.c 

.PHONY : clean 
clean : 
	-rm t $(tobj)  cscope.*

