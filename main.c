#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

struct matrix {
    int rows;
    int cols;
    int size; 
    float *entries;
};

struct neural_net {
    int num_layers;
    int *layers;
    struct matrix **weights;
};

float randf(float a, float b) {
    return ((float)rand()/(float)(RAND_MAX)) * (b - a) + a;
}

float rand_matrix_init() {
    return randf(-1.0, 1.0);
}

struct matrix *construct_matrix(int rows, int cols) {
    struct matrix *matrix = malloc(sizeof(struct matrix));

    matrix->rows = rows;
    matrix->cols = cols;
    matrix->size = rows * cols;

    matrix->entries = calloc(matrix->size, sizeof(float));
    return matrix;
}

struct matrix *destruct_matrix(struct matrix *matrix) {
    free(matrix->entries);
    free(matrix);
}

struct matrix *mat_mult(struct matrix *A, struct matrix *B) {
    struct matrix *mat_prod = construct_matrix(A->rows, B->cols);
    for (int row = 0; row < A->rows; ++row) {
        for (int col = 0; col < B->cols; ++col) {
            float prod = 0;
            float *start_row_A = A->entries + (row * A->cols);
            float *start_col_B = B->entries + col;
            for (int i = 0; i < A->cols; ++i) {
                prod += start_row_A[i] * start_col_B[B->cols * i]; 
            }
            (mat_prod->entries)[row * B->cols + col] = prod;
        }
    }
    return mat_prod;
}

struct neural_net *construct_neural_net(int num_layers, int layers[]) {
    struct neural_net *neural_net = malloc(sizeof(struct neural_net));
    neural_net->num_layers = num_layers;
    neural_net->layers = layers;
    neural_net->weights = malloc(num_layers * sizeof(struct matrix *));
    for (int layer = 0; layer < num_layers - 1; ++layer) {
        struct matrix *matrix = construct_matrix(layers[layer + 1], layers[layer]);
        for (int entry = 0; entry < matrix->size; ++entry) {
            (matrix->entries)[entry] = randf(-1.0,1.0);
        }
        (neural_net->weights)[layer] = matrix;
    }
    return neural_net;
}

void destruct_neural_net(struct neural_net *neural_net) {
    for (int layer = 0; layer < neural_net->num_layers - 1; ++layer) {
        destruct_matrix((neural_net->weights)[layer]);
    }
    free(neural_net);
}

void print_array(int size, int array[]) {
    printf("[");
    for(int i = 0; i < size; ++i) {
        if (i == size - 1) {
            printf("%d]", array[i]);
        } else {
            printf("%d, ", array[i]);
        }
    }
    return;
}

void print_matrix(struct matrix *matrix) {
    printf("[");
    for(int i = 0; i < matrix->size; ++i) {
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

int main() {
    int layers[] = {1,2,3};
    struct neural_net *neural_net = construct_neural_net(3, layers);
    printf("Num layers: %d\n", neural_net->num_layers);
    printf("Layers: ");
    print_array(neural_net->num_layers, neural_net->layers);
    printf("\n");
    printf("Weights: \n");
    for(int layer = 0; layer < neural_net->num_layers - 1; ++layer) {
        print_matrix((neural_net->weights)[layer]);
        printf("\n");
    }
    destruct_neural_net(neural_net);
    return 0;
}