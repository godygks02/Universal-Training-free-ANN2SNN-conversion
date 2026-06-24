#include "fpga_modules.h"

/**
 * IEEE 754 Exponent-Guided Bit-Slice Spiking Encoder.
 * 
 * Hardware characteristics:
 * - 100% Multiplier-free & Divider-free.
 * - Relies entirely on bitwise shifts, masks, and MUXs.
 * - This is the SINGLE top-level function for the Spiking Encoder hardware module.
 */
IEEE754_Spike_Enc ieee754_encoder_hls(float x, int timesteps) {
    #pragma HLS INLINE
    
    FloatInt val;
    val.f = x;
    uint32_t x_int = val.i;
    
    IEEE754_Spike_Enc enc;
    enc.S = (x_int >> 31) & 1;
    uint8_t E = (x_int >> 23) & 0xFF;
    uint32_t M = x_int & 0x7FFFFF;
    
    enc.e = (int16_t)E - 127;
    enc.spikes = 0;
    
    // Check absolute value or E==0 to handle zero/subnormal inputs
    FloatInt abs_val;
    abs_val.i = x_int & 0x7FFFFFFF;
    if (abs_val.f < 1e-15f || E == 0) {
        enc.spikes = 0;
        enc.e = -127;
        return enc;
    }
    
    // Temporal spike extraction loop
    // At t=1: Implicit leading bit (1 for normal numbers, 0 otherwise)
    // At t>=2: Mantissa bits starting from MSB
    for (int t = 1; t <= 16; t++) {
        #pragma HLS UNROLL
        if (t <= timesteps) {
            uint16_t spike_val = 0;
            if (t == 1) {
                spike_val = (E > 0) ? 1 : 0;
            } else {
                int shift = 24 - t;
                int safe_shift = (shift < 0) ? 0 : (shift > 22 ? 22 : shift);
                spike_val = (M >> safe_shift) & 1;
            }
            enc.spikes |= (spike_val << (16 - t));
        }
    }
    
    return enc;
}
