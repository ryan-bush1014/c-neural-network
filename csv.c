#include <stdio.h>
#include <stdlib.h>

int main()
{
    FILE *file;
    int character;
    file = fopen("mnist_test.csv", "r");
    printf("%d\n", (int) fgetc(file) - 48);
    fgetc(file);
    printf("%d\n", (int) fgetc(file) - 48);
    // while ((character = fgetc(file)) != EOF)
    // {
    //     printf((int) character);              // It displays the right character (UTF8) in the terminal
    // }
    return 0;
}