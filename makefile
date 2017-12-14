GCC = gcc -std=c99 -g -Wall
#CFLAGS = -I/usr/local/include/glib-2.0 -I/usr/local/lib/glib-2.0/include 
#CFLAGS += `pkg-config --cflags opencv`
#CLIBS += -lglib-2.0 -lpthread `pkg-config --libs opencv`   

tobj =  t.o  mnist.o
t: $(tobj)
	$(GCC)  -o t $(tobj)

mnist.o : mnist.h mnist.c 
t.o : t.c mnist.h mnist.c

.PHONY : clean 
clean : 
	-rm t $(tobj)

