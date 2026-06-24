#include <stdio.h>
#include <math.h>
#include "fpga_modules.h"

/**
 * IEEE 754 Spiking Decoder (Purely software-level testbench verification logic).
 * Kept inside the testbench to prevent synthesis of floating-point multipliers in the hardware IP.
 */
float ieee754_decoder_hls(IEEE754_Spike_Enc enc, int timesteps) {
    if (enc.e == -127) {
        return 0.0f;
    }
    
    float M_rec = 0.0f;
    
    for (int t = 1; t <= 16; t++) {
        if (t <= timesteps) {
            uint8_t spike = (enc.spikes >> (16 - t)) & 1;
            if (spike) {
                // Construct temporal weight d[t] = 2^(-t+1) using exponent field
                int16_t biased_d_e = -t + 1 + 127;
                if (biased_d_e > 0) {
                    FloatInt temp_d;
                    temp_d.i = ((uint32_t)biased_d_e << 23);
                    M_rec += temp_d.f;
                }
            }
        }
    }
    
    // Construct 2^e scale factor
    int16_t biased_e = enc.e + 127;
    if (biased_e <= 0) return 0.0f;
    if (biased_e >= 255) biased_e = 254;
    
    FloatInt scale_factor;
    scale_factor.i = ((uint32_t)biased_e << 23);
    
    // Apply sign factor (bitwise XOR)
    FloatInt final_val;
    final_val.f = M_rec * scale_factor.f;
    final_val.i ^= ((uint32_t)enc.S << 31);
    
    return final_val.f;
}

int main() {
    printf("====================================================================\n");
    printf("     Vitis HLS Testbench: IEEE 754 Spiking Encoder & Decoder\n");
    printf("====================================================================\n\n");
    
    float test_vals[5] = {0.875f, -0.3125f, 1.5f, -0.0001f, 0.0f};
    int timesteps = 16;
    int err_count = 0;
    
    for (int i = 0; i < 5; i++) {
        float x = test_vals[i];
        // Call top-level hardware encoder
        IEEE754_Spike_Enc enc = ieee754_encoder_hls(x, timesteps);
        // Call local software-level decoder
        float dec = ieee754_decoder_hls(enc, timesteps);
        float err = fabsf(x - dec);
        
        printf("  Original = %8.5f | Spikes = 0x%04X, Exp = %3d | Decoded = %8.5f | Rec Err = %8.6f\n",
               x, enc.spikes, enc.e, dec, err);
               
        if (fabsf(x) > 1e-4f && err > (1.0f / (1 << timesteps))) {
            printf("  [Error] Reconstruction error exceeds bit-slice limits!\n");
            err_count++;
        }
    }
    printf("\n");
    
    if (err_count == 0) {
        printf("====================================================================\n");
        printf("  [SUCCESS] IEEE 754 Spiking Encoder/Decoder C-Simulation passed.\n");
        printf("====================================================================\n");
        return 0;
    } else {
        printf("====================================================================\n");
        printf("  [FAILED] IEEE 754 Spiking Encoder/Decoder failed reconstruction!\n");
        printf("====================================================================\n");
        return 1;
    }
}
