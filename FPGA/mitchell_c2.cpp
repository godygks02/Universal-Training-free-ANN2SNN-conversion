#include "fpga_modules.h"

// Define Mitchell C-2 correction LUT (scaled by 2^23)
const int32_t MITCHELL_LUT[4][4] = {
    {130862, 393412, 655155, 917717},     // [0.015625, 0.046875, 0.078125, 0.109375] * 2^23
    {393412, 1179376, 1966276, 2752372},   // [0.046875, 0.140625, 0.234375, 0.328125] * 2^23
    {655155, 1966276, 3276636, 4587847},   // [0.078125, 0.234375, 0.390625, 0.546875] * 2^23
    {917717, 2752372, 4587847, 6422312}    // [0.109375, 0.328125, 0.546875, 0.765625] * 2^23
};

/**
 * Mitchell C-2 Logarithmic Multiplier.
 * Performs A * B using binary logarithmic addition with LUT-based piecewise correction.
 * 
 * Hardware properties:
 * - NO floating-point DSP multiplier blocks used.
 * - Resolves exponent addition, mantissa bit-shifts, and LUT indexing.
 * - Safe for HLS pipelining (Interval = 1 cycle).
 */
float mitchell_c2_multiply_hls(float A, float B) {
    #pragma HLS INLINE
    
    FloatInt val_A, val_B;
    val_A.f = A;
    val_B.f = B;
    
    uint32_t int_A = val_A.i;
    uint32_t int_B = val_B.i;
    
    // Extract Exponent bits (8 bits)
    uint8_t E_A = (int_A >> 23) & 0xFF;
    uint8_t E_B = (int_B >> 23) & 0xFF;
    
    // Zero-check: If E_A == 0 or E_B == 0, the input is zero or subnormal (underflow).
    // In deep learning applications, treating these as 0 is standard and hardware-friendly.
    if (E_A == 0 || E_B == 0) {
        return 0.0f;
    }
    
    // Extract Sign (1 bit) and Mantissa (23 bits)
    uint8_t S_A = (int_A >> 31) & 1;
    uint8_t S_B = (int_B >> 31) & 1;
    uint32_t M_A = int_A & 0x7FFFFF;
    uint32_t M_B = int_B & 0x7FFFFF;
    
    // Xor signs
    uint8_t S_out = S_A ^ S_B;
    
    // Look-Up Table (LUT) index calculation
    // Index = floor((M_val - 1.0) * 4.0).
    // Since M_val - 1.0 is fractional mantissa M_A, multiplying by 4.0 is equivalent to
    // shifting left by 2. Thus, the top 2 bits (bits 22 and 21) represent the index directly.
    uint8_t idx_A = (M_A >> 21) & 3;
    uint8_t idx_B = (M_B >> 21) & 3;
    
    int32_t C = MITCHELL_LUT[idx_A][idx_B];
    
    // Compute Mantissa Sum in Q1.23 representation
    // Formula: 1.M_A + 1.M_B - 1.0 + C = 1.0 + M_A + M_B + C
    // Sum is calculated in 23-bit fractional domain.
    int32_t M_sum = M_A + M_B + C;
    
    int16_t e_A = (int16_t)E_A - 127;
    int16_t e_B = (int16_t)E_B - 127;
    int16_t e_out = e_A + e_B;
    uint32_t M_out = 0;
    
    // Check for overflow (carryout of fractional sum >= 2.0, meaning >= 2^23)
    if (M_sum >= 8388608) { // 2^23
        M_out = (M_sum - 8388608) >> 1; // subtract 2.0 and divide by 2.0 (shift right by 1)
        e_out += 1;
    } else {
        M_out = M_sum;
    }
    
    // Re-bias exponent
    int16_t biased_E = e_out + 127;
    
    // Exponent range clamps (prevent overflow/underflow crash in float32 view)
    if (biased_E >= 255) {
        // Return infinity or max float
        biased_E = 254;
        M_out = 0x7FFFFF;
    } else if (biased_E <= 0) {
        // Return zero (underflow)
        return 0.0f;
    }
    
    uint8_t E_out = (uint8_t)biased_E;
    
    // Reassemble IEEE 754 float
    FloatInt val_out;
    val_out.i = ((uint32_t)S_out << 31) | ((uint32_t)E_out << 23) | (M_out & 0x7FFFFF);
    
    return val_out.f;
}
