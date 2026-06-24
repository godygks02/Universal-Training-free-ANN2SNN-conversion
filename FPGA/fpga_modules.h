#ifndef FPGA_MODULES_H
#define FPGA_MODULES_H

#include <stdint.h>

// Union for raw single-precision float bit manipulation
union FloatInt {
    float f;
    uint32_t i;
};

// ============================================================================
// Mitchell C-2 Logarithmic Multiplier Configuration
// ============================================================================
extern const int32_t MITCHELL_LUT[4][4];
float mitchell_c2_multiply_hls(float A, float B);

// ============================================================================
// IEEE 754 Exponent-Guided Spiking Encoder Configuration
// ============================================================================
struct IEEE754_Spike_Enc {
    uint8_t S;        // Sign bit (1-bit)
    int16_t e;       // Unbiased Exponent (e = E - 127)
    uint16_t spikes;  // Spikes representation: 16-bit mask
};

IEEE754_Spike_Enc ieee754_encoder_hls(float x, int timesteps);
float ieee754_decoder_hls(IEEE754_Spike_Enc enc, int timesteps);

// ============================================================================
// Spike-Driven S-PLA Activation Configuration
// ============================================================================
float spla_activation_hls(
    float x,
    float inv_scale_factor, // Replaced division with pre-computed reciprocal multiplication via Mitchell
    int timesteps,
    int prefix_k,
    int min_e_routing,
    const float* boundaries,
    const float* pre_scaled_slopes, // Pre-scaled slopes (a_i * scale_factor) to bypass runtime scaling
    const float* intercepts,
    int num_segments
);

#endif // FPGA_MODULES_H
