#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdio.h>

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
void destruct_matrix_array(int size, struct matrix **matrix_array);
struct matrix *copy_matrix(struct matrix *matrix);
struct matrix *mat_mult(struct matrix *A, struct matrix *B);
struct matrix *array_to_column(int size, float *arr);
struct matrix *transpose(struct matrix *matrix);
struct matrix *slice_row(struct matrix *matrix, int a, int b);
struct matrix *scale_matrix(struct matrix *matrix, float c);
struct matrix *binary_element_wise(struct matrix *A, struct matrix *B, float (*fptr)(float, float));
struct matrix *unary_element_wise(struct matrix *matrix, float (*fptr)(float));
struct matrix *matrix_add(struct matrix *A, struct matrix *B);
struct matrix *matrix_sub(struct matrix *A, struct matrix *B);
struct matrix *hadamard_product(struct matrix *A, struct matrix *B);
float squared_2_norm(struct matrix *matrix);

#endif // MATRIX_H