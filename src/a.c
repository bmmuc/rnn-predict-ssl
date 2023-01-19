#include <stdio.h>

int main() {
    char inpt1, inpt2;
    float price = 50;
    
    printf("Ele é estudante? \n");
    scanf("%c", &inpt1);

    printf("Ele é idoso? \n");
    scanf("%c", &inpt2);


    if(inpt1 == 'S') {
        price = price / 2.0 ;
    }

    else if(inpt2 == 'S') {
        price = price / 2.0;
    }

    else {
        price = price;
    }
}