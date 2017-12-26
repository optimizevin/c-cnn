CC = gcc
CFLAGS += -g -ansi -std=c99 -Wall -fgnu89-inline
CLIBS += -lm
#CLIBS += -lglib-2.0 -lpthread `pkg-config --libs opencv`   

tobj =  t.o  mnist.o cnn.o  nncomm.o
t: $(tobj)
	$(CC) -o t $(tobj) $(CLIBS)

nncomm.o: nncomm.h nncomm.c
mnist.o : mnist.h mnist.c 
t.o : t.c 
cnn.o : cnn.h cnn.c 

.PHONY : clean 
clean : 
	@rm -f t $(tobj)  cscope.*

