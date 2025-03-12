/*
 * matrix.c
 *
 * This file implements matrix operations, including construction, destruction,
 * printing, copying, multiplication, array conversion, and transposition.
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include "matrix.h" // This includes the definition of struct matrix

/*
 * print_array
 *
 * Prints the elements of an integer array to standard output.
 *
 * Parameters:
 * size: The number of elements in the array.
 * array: The integer array to be printed.
 *
 * Returns:
 * None.
 *
 * Side effects:
 * Prints the array elements to standard output.
 */
void print_array(int size, int array[])
{
    printf("[");
    for (int i = 0; i < size; ++i)
    {
        if (i == size - 1)
        {
            printf("%d]\n", array[i]);
        }
        else
        {
            printf("%d, ", array[i]);
        }
    }
    return;
}

/*
 * print_matrix
 *
 * Prints the elements of a matrix in row-major format to standard output.
 *
 * Parameters:
 * matrix: A pointer to the matrix to be printed.
 *
 * Returns:
 * None.
 *
 * Side effects:
 * Prints the matrix elements to standard output.
 */
void print_matrix(struct matrix *matrix)
{
    printf("[");
    for (int i = 0; i < matrix->size; ++i)
    {
        int col = i % matrix->cols;
        int row = i / matrix->cols;

        if (col == 0)
        {
            printf("[");
        }
        printf("%f", (matrix->entries)[i]);

        if (col == matrix->cols - 1)
        {
            printf("]");
            if (row < matrix->rows - 1)
            {
                printf("\n");
            }
        }
        else
        {
            printf(", ");
        }
    }
    printf("]\n");
    return;
}

/*
 * construct_matrix
 *
 * Constructs a matrix with the specified number of rows and columns.
 *
 * Parameters:
 * rows: The number of rows in the matrix.
 * cols: The number of columns in the matrix.
 *
 * Returns:
 * A pointer to the newly constructed matrix.
 *
 * Side effects:
 * Allocates memory for the matrix structure and its entries.
 */
struct matrix *construct_matrix(int rows, int cols)
{
    struct matrix *matrix = malloc(sizeof(struct matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->size = rows * cols;
    matrix->entries = calloc(matrix->size, sizeof(float));
    return matrix;
}

/*
 * destruct_matrix
 *
 * Deallocates the memory used by a matrix.
 *
 * Parameters:
 * matrix: A pointer to the matrix to be deallocated.
 *
 * Returns:
 * None.
 *
 * Side effects:
 * Deallocates memory for the matrix entries and the matrix structure.
 */
void destruct_matrix(struct matrix *matrix)
{
    free(matrix->entries);
    free(matrix);
}

void destruct_matrix_array(int size, struct matrix **matrix_array)
{
    for (int i = 0; i < size; ++i)
    {
        destruct_matrix(matrix_array[i]);
    }
    free(matrix_array);
}

/*
 * copy_matrix
 *
 * Constructs a copy of the given matrix.
 *
 * Parameters:
 * matrix: A pointer to the matrix to be copied.
 *
 * Returns:
 * A pointer to the newly constructed copy of the matrix.
 *
 * Side effects:
 * Allocates memory for the copied matrix and its entries.
 */
struct matrix *copy_matrix(struct matrix *matrix)
{
    struct matrix *copy = construct_matrix(matrix->rows, matrix->cols);
    for (int entry = 0; entry < matrix->size; ++entry)
    {
        (copy->entries)[entry] = (matrix->entries)[entry];
    }
    return copy;
}

/*
 * mat_mult
 *
 * Performs matrix multiplication of two matrices A and B.
 *
 * Parameters:
 * A: A pointer to the first matrix.
 * B: A pointer to the second matrix.
 * bias: A boolean value indicating whether to include bias in the multiplication.
 *
 * Returns:
 * A pointer to the resulting matrix from the multiplication.
 *
 * Side effects:
 * Allocates memory for the resulting matrix.
 */
struct matrix *mat_mult(struct matrix *A, struct matrix *B)
{
    assert(A->cols == B->rows);
    struct matrix *mat_prod = construct_matrix(A->rows, B->cols);
    for (int row = 0; row < A->rows; ++row)
    {
        for (int col = 0; col < B->cols; ++col)
        {
            float prod = 0;
            float *start_row_A = A->entries + (row * A->cols);
            float *start_col_B = B->entries + col;
            for (int i = 0; i < A->cols; ++i)
            {
                prod += start_row_A[i] * start_col_B[B->cols * i];
            }
            (mat_prod->entries)[row * B->cols + col] = prod;
        }
    }
    return mat_prod;
}

/*
 * array_to_column
 *
 * Converts an array to a column matrix.
 *
 * Parameters:
 * size: The number of elements in the array.
 * arr: The array to be converted.
 *
 * Returns:
 * A pointer to the resulting column matrix.
 *
 * Side effects:
 * Allocates memory for the column matrix.
 */
struct matrix *array_to_column(int size, float *arr)
{
    struct matrix *matrix = construct_matrix(size, 1);
    for (int entry = 0; entry < size; ++entry)
    {
        (matrix->entries)[entry] = arr[entry];
    }
    return matrix;
}

/*
 * transpose
 *
 * Transposes a matrix.
 *
 * Parameters:
 * matrix: A pointer to the matrix to be transposed.
 *
 * Returns:
 * A pointer to the transposed matrix.
 *
 * Side effects:
 * Allocates memory for the transposed matrix.
 */
struct matrix *transpose(struct matrix *matrix)
{
    struct matrix *transposed_matrix = construct_matrix(matrix->cols, matrix->rows);
    for (int row = 0; row < matrix->rows; ++row)
    {
        for (int col = 0; col < matrix->cols; ++col)
        {
            transposed_matrix->entries[col * transposed_matrix->cols + row] = matrix->entries[row * matrix->cols + col];
        }
    }
    return transposed_matrix;
}

struct matrix *slice_row(struct matrix *matrix, int a, int b)
{
    struct matrix *new = construct_matrix(b - a, matrix->cols);
    for (int i = 0; i < new->size; ++i)
    {
        new->entries[i] = matrix->entries[i + a * matrix->cols];
    }
    return new;
}

struct matrix *scale_matrix(struct matrix *matrix, float c)
{
    for (int entry = 0; entry < matrix->size; ++entry)
    {
        matrix->entries[entry] = matrix->entries[entry] * c;
    }
    return matrix;
}

struct matrix *binary_element_wise(struct matrix *A, struct matrix *B, float (*fptr)(float, float))
{
    assert((A->rows == B->rows) && (A->cols == B->cols));
    for (int entry = 0; entry < A->size; ++entry)
    {
        A->entries[entry] = fptr(A->entries[entry], B->entries[entry]);
    }
    return A;
}

float add(float a, float b)
{
    return a + b;
}

float sub(float a, float b)
{
    return a - b;
}

float mult(float a, float b)
{
    return a * b;
}

struct matrix *hadamard_product(struct matrix *A, struct matrix *B)
{
    return binary_element_wise(A, B, &mult);
}

struct matrix *matrix_add(struct matrix *A, struct matrix *B)
{
    return binary_element_wise(A, B, &add);
}

struct matrix *matrix_sub(struct matrix *A, struct matrix *B)
{
    return binary_element_wise(A, B, &sub);
}

float squared_2_norm(struct matrix *matrix)
{
    float sum = 0;
    for (int entry = 0; entry < matrix->size; ++entry)
    {
        sum += matrix->entries[entry] * matrix->entries[entry];
    }
    return sum;
}
