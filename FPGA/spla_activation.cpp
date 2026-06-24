#include "fpga_modules.h"

/**
 * Spike-Driven S-PLA Activation (GELU, Sigmoid, Tanh).
 * 
 * Hardware implementation details (100% Multiplier-free & Divider-free):
 * 1. Division 'x / scale_factor' is replaced by multiplying by pre-computed reciprocal 'inv_scale_factor' 
 *    using the hardware-wired Mitchell C-2 Multiplier: mitchell_c2_multiply_hls(x, inv_scale_factor).
 * 2. Prefix value reconstruction (x_rec_prefix) is performed via 0-latency bitwise reconstruction (M_int switch map).
 * 3. Segment lookup (bucketize) is a parallel comparator tree.
 * 4. a_i * scale_factor is pre-loaded as pre_scaled_slopes.
 * 5. Sign multiplication (* sign_factor) is replaced by bitwise XOR of the float sign bit (bit 31).
 */
float spla_activation_hls(
    float x,
    float inv_scale_factor, 
    int timesteps,
    int prefix_k,
    int min_e_routing,
    const float* boundaries,
    const float* pre_scaled_slopes, 
    const float* intercepts,
    int num_segments
) {
    // 1. Normalize input using Mitchell Logarithmic Multiplier instead of Float Divider
    float x_norm = mitchell_c2_multiply_hls(x, inv_scale_factor);
    if (x_norm > 1.0f) x_norm = 1.0f;
    if (x_norm < -1.0f) x_norm = -1.0f;
    
    // 2. Spiking encoding (Zero-cost shift bus)
    IEEE754_Spike_Enc enc = ieee754_encoder_hls(x_norm, timesteps);
    
    // 3. Bitwise Prefix Reconstitution (Multiplier-free)
    // Extract first K=3 spikes to build an integer M_int
    uint8_t M_int = 0;
    #pragma HLS INLINE
    if (prefix_k >= 1) M_int |= (((enc.spikes >> 15) & 1) << 2);
    if (prefix_k >= 2) M_int |= (((enc.spikes >> 14) & 1) << 1);
    if (prefix_k >= 3) M_int |= ((enc.spikes >> 13) & 1);
    
    // Map 3-bit spike pattern directly to mantissa and exponent offset
    // This replicates: M_rec = sum_t (s_x[t] * 2^(-t+1)) on [1, 2)
    int16_t dE = 0;
    uint32_t M_frac = 0;
    bool is_zero = false;
    
    switch (M_int) {
        case 0: is_zero = true; break;
        case 1: dE = 0; M_frac = 0; break;        // 1.0 * 2^0
        case 2: dE = 1; M_frac = 0; break;        // 1.0 * 2^1
        case 3: dE = 1; M_frac = 0x400000; break; // 1.5 * 2^1
        case 4: dE = 2; M_frac = 0; break;        // 1.0 * 2^2
        case 5: dE = 2; M_frac = 0x200000; break; // 1.25 * 2^2
        case 6: dE = 2; M_frac = 0x400000; break; // 1.5 * 2^2
        case 7: dE = 2; M_frac = 0x600000; break; // 1.75 * 2^2
        default: is_zero = true; break;
    }
    
    float x_rec_prefix = 0.0f;
    if (!is_zero && enc.e >= min_e_routing) {
        int16_t e_prefix = (enc.e < min_e_routing) ? min_e_routing : enc.e;
        // Total power: e_prefix - 2 (since M_int is scaled by 2^2) + dE
        int16_t biased_E = e_prefix - 2 + dE + 127;
        
        if (biased_E > 0 && biased_E < 255) {
            FloatInt temp_prefix;
            // Assemble float directly using bitwise OR
            temp_prefix.i = ((uint32_t)enc.S << 31) | ((uint32_t)biased_E << 23) | M_frac;
            x_rec_prefix = temp_prefix.f;
        }
    }
    
    // 4. Segment Lookup (Bucketize)
    int seg_idx = 0;
    for (int i = 0; i < num_segments - 1; i++) {
        #pragma HLS UNROLL
        if (x_rec_prefix >= boundaries[i]) {
            seg_idx = i + 1;
        }
    }
    
    // Fetch pre-scaled slope A_base = a_i * scale_factor and intercept
    float A_base = pre_scaled_slopes[seg_idx];
    float b_i = intercepts[seg_idx];
    
    FloatInt temp_base;
    temp_base.f = A_base;
    uint32_t int_base = temp_base.i;
    uint8_t E_base = (int_base >> 23) & 0xFF;
    
    float term_sum = 0.0f;
    
    // 5. Exponent-Shift-and-Add (No floating-point multipliers)
    if (E_base > 0) {
        for (int t = 1; t <= 16; t++) {
            #pragma HLS UNROLL
            if (t <= timesteps) {
                uint8_t spike = (enc.spikes >> (16 - t)) & 1;
                if (spike) {
                    // Shift A_base by 2^(e - t + 1) using integer exponent addition
                    int16_t E_new = (int16_t)E_base + enc.e - t + 1;
                    
                    FloatInt shifted_val;
                    if (E_new <= 0) {
                        shifted_val.f = 0.0f;
                    } else if (E_new >= 255) {
                        shifted_val.i = (int_base & 0x807FFFFF) | (254 << 23);
                    } else {
                        shifted_val.i = (int_base & 0x807FFFFF) | ((uint32_t)E_new << 23);
                    }
                    
                    // Apply sign (XOR bit 31 directly to bypass float multiplication)
                    shifted_val.i ^= ((uint32_t)enc.S << 31);
                    term_sum += shifted_val.f; // Float Accumulator
                }
            }
        }
    }
    
    // Output: f(x) = a_i * x + b_i
    return b_i + term_sum;
}
