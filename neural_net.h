#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

struct neural_net {
    int num_layers;
    int *layers;
    struct matrix **weights;
    struct matrix **biases;
};

// Function declarations
void print_neural_net(struct neural_net *neural_net);
float randf(float a, float b);
struct neural_net *construct_neural_net(int num_layers, int layers[]);
void destruct_neural_net(struct neural_net *neural_net);
struct matrix *eval(struct neural_net *neural_net, struct matrix *in_data);
float back_propagate(struct neural_net *neural_net, struct matrix *in_data, struct matrix *expected, float learning_rate);

#endif // NEURAL_NET_H
