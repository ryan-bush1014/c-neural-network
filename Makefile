CC = gcc
CFLAGS = -Wall
LDFLAGS = -O3 -lm  # Link the math library

# Target to create the final executable
my_program: main.o matrix.o neural_net.o
	$(CC) -o my_program main.o matrix.o neural_net.o $(LDFLAGS)

# Rule to compile main.o
main.o: main.c matrix.h neural_net.h
	$(CC) -c main.c $(CFLAGS)

# Rule to compile matrix.o
matrix.o: matrix.c matrix.h
	$(CC) -c matrix.c $(CFLAGS)

# Rule to compile neural_net.o
neural_net.o: neural_net.c neural_net.h
	$(CC) -c neural_net.c $(CFLAGS)

# Clean up generated files
clean:
	rm -f *.o my_program
