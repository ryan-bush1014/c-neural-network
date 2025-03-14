#ifndef NEURAL_NET_H
#define NEURAL_NET_H

struct neural_net {
    int num_layers;
    int *layers;
    struct matrix **weights;
    struct matrix **biases;
    float (*(*activations))(float);
    float (*(*activations_derivatives))(float);
};

// Function declarations
void print_neural_net(struct neural_net *neural_net);
float randf(float a, float b);
struct neural_net *construct_neural_net(int num_layers, int layers[], char **activations);
void destruct_neural_net(struct neural_net *neural_net);
struct matrix *eval(struct neural_net *neural_net, struct matrix *in_data);
float back_propagate(struct neural_net *neural_net, struct matrix *in_data, struct matrix *expected, float learning_rate);

#endif // NEURAL_NET_H
