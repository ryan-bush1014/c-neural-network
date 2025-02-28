#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "neural_net.h"

void print_neural_net(struct neural_net *neural_net) {
    printf("Num layers: %d\n\n", neural_net->num_layers);
    printf("Layers: ");
    print_array(neural_net->num_layers, neural_net->layers);
    printf("\nWeights:\n");
    for(int layer = 0; layer < neural_net->num_layers - 1; ++layer) {
        print_matrix((neural_net->weights)[layer]);
        printf("\n");
    }
}

float randf(float a, float b) {
    return ((float)rand()/(float)(RAND_MAX)) * (b - a) + a;
}

struct neural_net *construct_neural_net(int num_layers, int layers[]) {
    struct neural_net *neural_net = malloc(sizeof(struct neural_net));
    neural_net->num_layers = num_layers;
    neural_net->layers = layers;
    neural_net->weights = malloc(num_layers * sizeof(struct matrix *));
    for (int layer = 0; layer < num_layers - 1; ++layer) {
        struct matrix *matrix = construct_matrix(layers[layer + 1], layers[layer] + 1);
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
    free(neural_net->weights);
    free(neural_net);
}

void sigmoid(struct matrix *output) {
    for (int i = 0; i < output->size; ++i) {
        output->entries[i] = 1.0f / (1.0f + expf(-output->entries[i]));
    }
}

void sigmoid_derivative(struct matrix *output) {
    for (int i = 0; i < output->size; ++i) {
        float y = output->entries[i];
        output->entries[i] = y * (1.0f - y);
    }
}

struct matrix *eval(struct neural_net *neural_net, struct matrix *in_data) {
    struct matrix *current = copy_matrix(in_data);
    for (int layer = 0; layer < neural_net->num_layers - 1; ++layer) {
        struct matrix *old = current;
        current = mat_mult((neural_net->weights)[layer], current, true);
        sigmoid(current);
        destruct_matrix(old);
    }
    return current;
}