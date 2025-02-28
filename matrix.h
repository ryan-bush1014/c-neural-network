#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

struct matrix {
    int rows;
    int cols;
    int size;
    float *entries;
};

// Function declarations
void print_array(int size, int array[]);
void print_matrix(struct matrix *matrix);
struct matrix *construct_matrix(int rows, int cols);
void destruct_matrix(struct matrix *matrix);
struct matrix *copy_matrix(struct matrix *matrix);
struct matrix *mat_mult(struct matrix *A, struct matrix *B, bool bias);
struct matrix *array_to_column(int size, float *arr);
struct matrix *transpose(struct matrix *matrix);

#endif // MATRIX_H