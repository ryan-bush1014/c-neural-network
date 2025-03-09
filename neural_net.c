/*
 * neural_net.c
 *
 * This file implements neural network operations, including construction,
 * destruction, printing, evaluation, and activation functions.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "neural_net.h"

/*
 * print_neural_net
 *
 * Prints the structure and weights of a neural network to standard output.
 *
 * Parameters:
 * neural_net: A pointer to the neural network to be printed.
 *
 * Returns:
 * None.
 *
 * Side effects:
 * Prints the neural network structure and weights to standard output.
 */
void print_neural_net(struct neural_net *neural_net)
{
    printf("Num layers: %d\n\n", neural_net->num_layers);
    printf("Layers: ");
    print_array(neural_net->num_layers, neural_net->layers);
    printf("\nWeights:\n");
    for (int layer = 0; layer < neural_net->num_layers - 1; ++layer)
    {
        print_matrix((neural_net->weights)[layer]);
        printf("\n");
    }

    printf("\nBiases:\n");
    for (int layer = 0; layer < neural_net->num_layers - 1; ++layer)
    {
        print_matrix((neural_net->biases)[layer]);
        printf("\n");
    }
}

/*
 * randf
 *
 * Generates a random float number between a and b.
 *
 * Parameters:
 * a: The minimum value of the range.
 * b: The maximum value of the range.
 *
 * Returns:
 * A random float number between a and b.
 *
 * Side effects:
 * None.
 */
float randf(float a, float b)
{
    return ((float)rand() / (float)(RAND_MAX)) * (b - a) + a;
}

/*
 * construct_neural_net
 *
 * Constructs a neural network with the specified number of layers and layer sizes.
 *
 * Parameters:
 * num_layers: The number of layers in the neural network.
 * layers: An array of layer sizes.
 *
 * Returns:
 * A pointer to the newly constructed neural network.
 *
 * Side effects:
 * Allocates memory for the neural network structure and its weights.
 */
struct neural_net *construct_neural_net(int num_layers, int layers[])
{
    struct neural_net *neural_net = malloc(sizeof(struct neural_net));
    neural_net->num_layers = num_layers;
    neural_net->layers = layers;
    neural_net->weights = malloc((num_layers - 1) * sizeof(struct matrix *));
    neural_net->biases = malloc((num_layers - 1) * sizeof(float *));
    for (int layer = 0; layer < num_layers - 1; ++layer)
    {
        struct matrix *matrix = construct_matrix(layers[layer + 1], layers[layer]);
        struct matrix *bias = construct_matrix(layers[layer + 1], 1);
        for (int entry = 0; entry < matrix->size; ++entry)
        {
            matrix->entries[entry] = randf(-0.5f, 0.5f);
        }
        neural_net->weights[layer] = matrix;
        neural_net->biases[layer] = bias;
    }
    return neural_net;
}

/*
 * destruct_neural_net
 *
 * Deallocates the memory used by a neural network.
 *
 * Parameters:
 * neural_net: A pointer to the neural network to be deallocated.
 *
 * Returns:
 * None.
 *
 * Side effects:
 * Deallocates memory for the neural network weights and the neural network structure.
 */
void destruct_neural_net(struct neural_net *neural_net)
{
    destruct_matrix_array(neural_net->num_layers - 1, neural_net->weights);
    free(neural_net);
}

/*
 * sigmoid
 *
 * Applies the sigmoid function to each element of a matrix.
 *
 * Parameters:
 * output: A pointer to the matrix to which the sigmoid function will be applied.
 *
 * Returns:
 * None.
 *
 * Side effects:
 * Modifies the matrix entries in place.
 */
struct matrix *sigmoid(struct matrix *matrix)
{
    for (int i = 0; i < matrix->size; ++i)
    {
        matrix->entries[i] = 1.0f / (1.0f + expf(-matrix->entries[i]));
    }
    return matrix;
}

/*
 * sigmoid_derivative
 *
 * Applies the derivative of the sigmoid function to each element of a matrix.
 *
 * Parameters:
 * output: A pointer to the matrix to which the derivative will be applied.
 *
 * Returns:
 * None.
 *
 * Side effects:
 * Modifies the matrix entries in place.
 */
struct matrix *sigmoid_derivative(struct matrix *matrix)
{
    for (int i = 0; i < matrix->size; ++i)
    {
        matrix->entries[i] = 1.0f / (1.0f + expf(-matrix->entries[i])) * (1.0f - 1.0f / (1.0f + expf(-matrix->entries[i])));
    }
    return matrix;
}

/*
 * eval
 *
 * Evaluates the neural network with the given input data.
 *
 * Parameters:
 * neural_net: A pointer to the neural network.
 * in_data: A pointer to the input data matrix.
 *
 * Returns:
 * A pointer to the output matrix.
 *
 * Side effects:
 * Allocates and deallocates memory for intermediate matrices.
 */
struct matrix *eval(struct neural_net *neural_net, struct matrix *in_data)
{
    struct matrix *current = copy_matrix(in_data);
    for (int layer = 0; layer < neural_net->num_layers - 1; ++layer)
    {
        struct matrix *old = current;
        current = mat_mult((neural_net->weights)[layer], current);
        for (int col = 0; col < current->cols; ++col)
        {
            for (int row = 0; row < current->rows; ++row)
            {
                current->entries[row * current->cols + col] += (((neural_net->biases)[layer])->entries)[row];
            }
        }
        sigmoid(current);
        destruct_matrix(old);
    }
    return current;
}

float back_propagate(struct neural_net *neural_net, struct matrix *in_data, struct matrix *expected, float learning_rate)
{
    struct matrix **Z = calloc(neural_net->num_layers - 1, sizeof(struct matrix *));
    struct matrix **activations = calloc(neural_net->num_layers, sizeof(struct matrix *));
    activations[0] = copy_matrix(in_data);

    // Forward pass
    for (int layer = 0; layer < neural_net->num_layers - 1; ++layer)
    {
        Z[layer] = mat_mult((neural_net->weights)[layer], activations[layer]);
        for (int col = 0; col < Z[layer]->cols; ++col)
        {
            for (int row = 0; row < Z[layer]->rows; ++row)
            {
                Z[layer]->entries[row * Z[layer]->cols + col] += (((neural_net->biases)[layer])->entries)[row];
            }
        }

        activations[layer + 1] = sigmoid(copy_matrix(Z[layer]));
    }

    struct matrix **dCdA = calloc(neural_net->num_layers - 1, sizeof(struct matrix *));
    struct matrix **dCdZ = calloc(neural_net->num_layers - 1, sizeof(struct matrix *));

    dCdA[neural_net->num_layers - 2] = matrix_sub(copy_matrix(activations[neural_net->num_layers - 1]), expected);
    for (int layer = neural_net->num_layers - 2; layer >= 0; --layer)
    {
        struct matrix *dAdZ = sigmoid_derivative(copy_matrix(Z[layer]));
        dCdZ[layer] = hadamard_product(copy_matrix(dCdA[layer]), dAdZ);

        struct matrix *dZdW = copy_matrix(activations[layer]);
        struct matrix *dZdW_transposed = transpose(dZdW);

        struct matrix *dCdW = scale_matrix(mat_mult(dCdZ[layer], dZdW_transposed), learning_rate);
        matrix_sub(neural_net->weights[layer], dCdW);

        struct matrix *ones = construct_matrix(dCdZ[layer]->cols, 1);
        for (size_t entry = 0; entry < ones->size; entry++)
        {
            ones->entries[entry] = 1.0f;
        }
        struct matrix *dCdB = scale_matrix(mat_mult(dCdZ[layer], ones), learning_rate);
        matrix_sub(neural_net->biases[layer], dCdB);

        destruct_matrix(dCdB);
        destruct_matrix(ones);
        destruct_matrix(dCdW);
        destruct_matrix(dZdW_transposed);
        destruct_matrix(dZdW);
        destruct_matrix(dAdZ);

        if (layer != 0)
        {
            struct matrix *dZdA = copy_matrix(neural_net->weights[layer]);
            struct matrix *dZdA_transposed = transpose(dZdA);
            dCdA[layer - 1] = mat_mult(dZdA_transposed, dCdZ[layer]);

            destruct_matrix(dZdA);
            destruct_matrix(dZdA_transposed);
        }
    }

    float cost = 0.5f * squared_2_norm(dCdA[neural_net->num_layers - 2]);

    destruct_matrix_array(neural_net->num_layers - 1, dCdA);
    destruct_matrix_array(neural_net->num_layers - 1, dCdZ);

    return cost;
}

//  // Compute output error
//  struct matrix *dCdA = matrix_sub(copy_matrix(activations[neural_net->num_layers - 1]), expected);

//  // all of the following works as long as I have the current layers dCdA
//  struct matrix *dAdZ = sigmoid_derivative(copy_matrix(Z[neural_net->num_layers - 2]));
//  struct matrix *dCdZ = hadamard_product(copy_matrix(dCdA), dAdZ);
//  // dZij/dWkl = 0 if k neq i, A-1lj
//  struct matrix *dZdW = copy_matrix(activations[neural_net->num_layers - 2]);
//  struct matrix *dZdW_transposed = transpose(dZdW);
//  struct matrix *dCdW = mat_mult(dCdZ, dZdW_transposed);
//  // struct matrix *dCdB = mat_mult(dCdZ, dZdW_transposed);
//  // dCdB_i = sum_l dCdZil
//  struct matrix *ones = construct_matrix(dCdZ->rows, 1);
//  for (size_t entry = 0; entry < ones->size; entry++)
//  {
//      ones->entries[entry] = 1.0f;
//  }
//  struct matrix *dCdB = mat_mult(dCdZ, ones);

//  // across layers we have dCdZ, dCdA

//  //                  | we know    | we don't know
//  // dCdAij = transpose(W) dCdZ
//  // dZ+1kjdAij = Wki
//  struct matrix *dZdA = copy_matrix(neural_net->weights[neural_net->num_layers - 2]);
//  struct matrix *dZdA_transposed = transpose(dZdA);
//  struct matrix *dCdA = mat_mult(dZdA_transposed, dCdZ);
