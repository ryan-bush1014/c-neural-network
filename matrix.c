#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "matrix.h"  // This includes the definition of struct matrix

void print_array(int size, int array[]) {
    printf("[");
    for (int i = 0; i < size; ++i) {
        if (i == size - 1) {
            printf("%d]\n", array[i]);
        } else {
            printf("%d, ", array[i]);
        }
    }
    return;
}

void print_matrix(struct matrix *matrix) {
    printf("[");
    for (int i = 0; i < matrix->size; ++i) {
        int col = i % matrix->cols;
        int row = i / matrix->cols;

        if (col == 0) {
            printf("[");
        }
        printf("%f", (matrix->entries)[i]);

        if (col == matrix->cols - 1) {
            printf("]");
            if (row < matrix->rows - 1) {
                printf("\n");
            }
        } else {
            printf(", ");
        }
    }
    printf("]\n");
    return;
}

struct matrix *construct_matrix(int rows, int cols) {
    struct matrix *matrix = malloc(sizeof(struct matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->size = rows * cols;
    matrix->entries = calloc(matrix->size, sizeof(float));
    return matrix;
}

void destruct_matrix(struct matrix *matrix) {
    free(matrix->entries);
    free(matrix);
}

struct matrix *copy_matrix(struct matrix *matrix) {
    struct matrix *copy = construct_matrix(matrix->rows, matrix->cols);
    for (int entry = 0; entry < matrix->size; ++entry) {
        (copy->entries)[entry] = (matrix->entries)[entry];
    }
    return copy;
}

struct matrix *mat_mult(struct matrix *A, struct matrix *B, bool bias) {
    struct matrix *mat_prod = construct_matrix(A->rows, B->cols);
    for (int row = 0; row < A->rows; ++row) {
        for (int col = 0; col < B->cols; ++col) {
            float prod = 0;
            float *start_row_A = A->entries + (row * A->cols);
            float *start_col_B = B->entries + col;
            for (int i = 0; i < A->cols - (bias ? 1 : 0); ++i) {
                prod += start_row_A[i] * start_col_B[B->cols * i];
            }
            prod += bias ? start_row_A[A->cols - 1] : 0;
            (mat_prod->entries)[row * B->cols + col] = prod;
        }
    }
    return mat_prod;
}

struct matrix *array_to_column(int size, float *arr) {
    struct matrix *matrix = construct_matrix(size, 1);
    for (int entry = 0; entry < size; ++entry) {
        (matrix->entries)[entry] = arr[entry];
    }
    return matrix;
}

struct matrix *transpose(struct matrix *matrix) {
    struct matrix *transposed_matrix = construct_matrix(matrix->cols, matrix->rows);
    for (int row = 0; row < matrix->rows; ++row) {
        for (int col = 0; col < matrix->cols; ++col) {
            transposed_matrix->entries[col * transposed_matrix->cols + row] = matrix->entries[row * matrix->cols + col];
        }
    }
    return transposed_matrix;
}