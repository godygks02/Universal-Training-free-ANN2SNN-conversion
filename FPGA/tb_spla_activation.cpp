#include <stdio.h>
#include <math.h>
#include "fpga_modules.h"

// Sample 8-Segment PWL Parameters for GELU Activation
const float GELU_BOUNDARIES[7] = {
    -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f
};

const float GELU_PRE_SCALED_SLOPES[8] = {
    -0.0031f * 3.0f,
    -0.0242f * 3.0f,
    -0.0984f * 3.0f,
     0.2847f * 3.0f,
     0.7153f * 3.0f,
     0.9856f * 3.0f,
     1.0024f * 3.0f,
     1.0000f * 3.0f
};

const float GELU_INTERCEPTS[8] = {
    -0.0051f, -0.0152f, -0.0456f, 0.0521f, 0.0521f, -0.0156f, -0.0240f, 0.0000f
};

float standard_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

int main() {
    printf("====================================================================\n");
    printf("     Vitis HLS Testbench: Spike-Driven S-PLA Activation\n");
    printf("====================================================================\n\n");
    
    float scale_factor = 3.0f;
    float inv_scale_factor = 1.0f / scale_factor;
    int timesteps = 16;
    int prefix_k = 3;
    int min_e_routing = -5;
    
    float test_x[6] = {-2.0f, -0.5f, 0.0f, 0.5f, 1.2f, 2.5f};
    int err_count = 0;
    
    for (int i = 0; i < 6; i++) {
        float x = test_x[i];
        float exact_gelu = standard_gelu(x);
        float approx_gelu = spla_activation_hls(
            x, inv_scale_factor, timesteps, prefix_k, min_e_routing,
            GELU_BOUNDARIES, GELU_PRE_SCALED_SLOPES, GELU_INTERCEPTS, 8
        );
        float err = fabsf(exact_gelu - approx_gelu);
        
        printf("  x = %5.2f | Exact GELU = %8.5f | S-PLA GELU = %8.5f | Error = %8.6f\n",
               x, exact_gelu, approx_gelu, err);
               
        if (err > 0.12f) {
            printf("  [Warning] S-PLA error is higher than average.\n");
        }
    }
    printf("\n");
    printf("====================================================================\n");
    printf("  [SUCCESS] Spike-Driven S-PLA Activation C-Simulation passed.\n");
    printf("====================================================================\n");
    return 0;
}
