CC=gcc
CFLAGS=-fPIC -I.

haarrandom.o: haarrandom.c
	$(CC) $(CFLAGS) -c -o $@ $^

haarrandom.a: haarrandom.o
	ar rcs haarrandom.a haarrandom.o

clean:
	rm -f haarrandom.a haarrandom.o
	rm -f *.so

.PHONY: all
all: haarrandom.a
