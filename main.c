#include "matrix.h"
#include "neural_net.h"

int main() {
    int layers[] = {1, 1};
    struct neural_net *neural_net = construct_neural_net(2, layers);
    print_neural_net(neural_net);

    float in_arr[] = {2};
    printf("Input:\n");
    struct matrix *in_matrix = array_to_column(1,in_arr);
    print_matrix(in_matrix);
    printf("\nOutput:\n");
    print_matrix(eval(neural_net, in_matrix));

    destruct_neural_net(neural_net);
    return 0;
}