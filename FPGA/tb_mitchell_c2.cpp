#include <stdio.h>
#include <math.h>
#include "fpga_modules.h"

int main() {
    printf("====================================================================\n");
    printf("     Vitis HLS Testbench: Mitchell C-2 Logarithmic Multiplier\n");
    printf("====================================================================\n\n");
    
    float test_pairs[5][2] = {
        {1.5f, 2.3f},
        {-0.75f, 4.0f},
        {12.5f, -0.5f},
        {0.0f, 9.87f},
        {-3.14f, -2.71f}
    };
    
    for (int i = 0; i < 5; i++) {
        float A = test_pairs[i][0];
        float B = test_pairs[i][1];
        float exact = A * B;
        float approx = mitchell_c2_multiply_hls(A, B);
        float err = fabsf(exact - approx);
        float rel_err = (exact != 0.0f) ? (err / fabsf(exact)) * 100.0f : 0.0f;
        
        printf("  A = %6.2f, B = %6.2f | Exact = %8.4f, Approx = %8.4f | Error = %6.4f (%5.2f%%)\n",
               A, B, exact, approx, err, rel_err);
               
        if (exact != 0.0f && rel_err > 6.0f) {
            printf("  [Warning] Relative error is higher than average.\n");
        }
    }
    printf("\n");
    printf("====================================================================\n");
    printf("  [SUCCESS] Mitchell C-2 Multiplier C-Simulation check completed.\n");
    printf("====================================================================\n");
    return 0;
}
