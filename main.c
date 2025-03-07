#include "matrix.h"
#include "neural_net.h"

int main() {
    int layers[] = {2, 3, 1};
    struct neural_net *neural_net = construct_neural_net(3, layers);

    struct matrix *in_matrix = construct_matrix(2, 4);
    in_matrix->entries[0] = 0.0f;
    in_matrix->entries[4] = 0.0f;
    in_matrix->entries[1] = 0.0f;
    in_matrix->entries[5] = 1.0f;
    in_matrix->entries[2] = 1.0f;
    in_matrix->entries[6] = 0.0f;
    in_matrix->entries[3] = 1.0f;
    in_matrix->entries[7] = 1.0f;

    struct matrix *expected = construct_matrix(1, 4);
    expected->entries[0] = 0.0f;
    expected->entries[1] = 1.0f;
    expected->entries[2] = 1.0f;
    expected->entries[3] = 0.0f;

    for (size_t i = 0; i < 1000000; i++)
    {
        back_propagate(neural_net, in_matrix, expected, 0.5f);
        // printf("Cost: %f\n\n", back_propagate(neural_net, in_matrix, expected, 0.5f));
    }

    print_matrix(eval(neural_net, in_matrix));

    destruct_neural_net(neural_net);
    destruct_matrix(in_matrix);
    return 0;
}