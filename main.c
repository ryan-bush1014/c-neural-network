#include "matrix.h"
#include "neural_net.h"
#include <string.h>

float *read_csv(char *csv, int size)
{
    float *data = calloc(size, sizeof(int));
    FILE *file;
    file = fopen(csv, "r");
    int current_num = 0;
    while (current_num < size)
    {
        int character = fgetc(file);
        if (character < 48 || character > 57)
        {
            --current_num;
        }
        while (48 <= character && character <= 57)
        {
            data[current_num] = data[current_num] * 10 + ((float)(character - 48));
            character = fgetc(file);
        }
        data[current_num] = data[current_num];
        ++current_num;
    }
    return data;
}

void get_batches(char *csv, int batch_size, int rows, int cols, struct matrix ***inputs, struct matrix ***outputs)
{
    int batches = rows / batch_size;
    float *data = read_csv(csv, rows * cols);
    *inputs = malloc(batches * sizeof(struct matrix *));
    *outputs = malloc(batches * sizeof(struct matrix *));

    int batch_size_flat = batch_size * cols;
    for (int batch = 0; batch < batches; ++batch)
    {
        struct matrix batch_matrix = {0};
        batch_matrix.rows = batch_size;
        batch_matrix.cols = cols;
        batch_matrix.size = batch_size_flat;
        batch_matrix.entries = data + (batch * batch_size_flat);

        struct matrix *batch_transposed = transpose(&batch_matrix);
        (*inputs)[batch] = scale_matrix(slice_row(batch_transposed, 1, batch_transposed->rows), 1.0f / 255.0f);

        destruct_matrix(batch_transposed);

        struct matrix *output = construct_matrix(10, batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            output->entries[((int)data[(batch * batch_size + i) * cols]) * output->cols + i] = 1.0f;
        }

        (*outputs)[batch] = output;
    }
    free(data);
}

float test_accuracy(struct neural_net *neural_net, struct matrix *input_test, struct matrix *output_test)
{
    struct matrix *test = eval(neural_net, input_test);
    int correct = 0;
    for (int col = 0; col < output_test->cols; ++col)
    {
        float max_real = output_test->entries[col];
        int max_index_real = 0;
        float max_net = test->entries[col];
        int max_index_net = 0;
        for (int row = 1; row < 10; ++row)
        {
            if (output_test->entries[row * output_test->cols + col] > max_real)
            {
                max_index_real = row;
                max_real = output_test->entries[row * output_test->cols + col];
            }
            if (test->entries[row * test->cols + col] > max_net)
            {
                max_index_net = row;
                max_net = output_test->entries[row * output_test->cols + col];
            }
        }
        if (max_index_real == max_index_net)
        {
            ++correct;
        }
    }
    destruct_matrix(test);
    return ((float)correct) / ((float)output_test->cols);
}

void display_mnist_image(struct matrix *images, int col)
{
    char shades[] = " .:-=+*#%@"; // ASCII intensity mapping

    for (int y = 0; y < 28; ++y)
    {
        for (int x = 0; x < 28; ++x)
        {
            float pixel = images->entries[col + (y * 28 + x) * images->cols]; // Get pixel intensity (0-1)
            int shade_index = (int)(pixel * 9);                               // Map to ASCII shades
            putchar(shades[shade_index]);
            putchar(' '); // Add spacing for readability
        }
        putchar('\n');
    }
}

int main()
{
    int layers[] = {784, 10, 10, 10};
    struct neural_net *neural_net = construct_neural_net(4, layers);

    int batch_size = 96;
    int rows_train = 60000;
    int rows_test = 10000;
    int cols = 28 * 28 + 1;
    int batches = rows_train / batch_size;
    struct matrix **inputs_train = NULL;
    struct matrix **outputs_train = NULL;
    struct matrix **inputs_test = NULL;
    struct matrix **outputs_test = NULL;

    printf("Loading data from persistent storage...\n");

    get_batches("mnist_train.csv", batch_size, rows_train, cols, &inputs_train, &outputs_train);
    get_batches("mnist_test.csv", rows_test, rows_test, cols, &inputs_test, &outputs_test);

    printf("Data loaded. Training...\n");

    int epochs = 5;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        float cost = 0;
        for (int batch = 0; batch < batches; ++batch)
        {
            cost += back_propagate(neural_net, inputs_train[batch], outputs_train[batch], 0.05f);
        }
        printf("Epoch %d - Cost: %f, Accuracy: %f%%\n", epoch, cost, test_accuracy(neural_net, inputs_test[0], outputs_test[0]) * 100.0f);
    }

    printf("Training completed. Testing...\n");

    int i = 0;
    struct matrix *out = eval(neural_net, inputs_test[0]);
    while (getchar() != 'e')
    {
        display_mnist_image(inputs_test[0], i);
        for (int row = 0; row < out->rows; ++row)
        {
            printf("%f ", out->entries[i + row * out->cols]);
        }
        putchar('\n');
        ++i;
    }
    destruct_matrix(out);

    destruct_matrix_array(batches, inputs_train);
    destruct_matrix_array(batches, outputs_train);
    destruct_matrix_array(1, inputs_test);
    destruct_matrix_array(1, outputs_test);
    destruct_neural_net(neural_net);
    return 0;
}