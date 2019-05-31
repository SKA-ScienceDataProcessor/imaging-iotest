
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <fcntl.h>
#include <stdint.h>
#include <complex.h>
#include <fftw3.h>
#include <omp.h>
#include <string.h>
#include <fenv.h>
#ifdef __SSE4_1__
#include <immintrin.h>
#endif

#include "grid.h"

// Assume all uvw are zero. This eliminates all coordinate
// calculations and grids only into the middle.
//#define ASSUME_UVW_0

inline static int coord(int grid_size, double theta,
                        struct bl_data *bl_data,
                        int time, int freq) {
#ifdef ASSUME_UVW_0
    int x = 0, y = 0;
#else
    int x = (int)floor(theta * uvw_lambda(bl_data, time, freq, 0) + .5);
    int y = (int)floor(theta * uvw_lambda(bl_data, time, freq, 1) + .5);
#endif
    return (y+grid_size/2) * grid_size + (x+grid_size/2);
}

inline static void _frac_coord(int grid_size, int oversample,
                               double u, int *x, int *fx) {

    // Round to nearest oversampling value. We assume the kernel to be
    // in "natural" order, so what we are looking for is the best
    //    "x - fx / oversample"
    // approximation for "grid_size/2 + u".
    double o = (grid_size / 2 - u) * oversample;
#ifdef __SSE4_1__
    int ox = _mm_round_pd(_mm_set_pd(0, o),_MM_FROUND_TO_NEAREST_INT)[0];
#else
    fesetround(3); int ox = lrint(o);
#endif
    *x = grid_size-(ox / oversample);
    *fx = ox % oversample;

}

void frac_coord(int grid_size, int oversample,
                double u, int *x, int *fx) {

    _frac_coord(grid_size, oversample, u, x, fx);

}

// Fractional coordinate calculation for separable 1D kernel
inline static void frac_coord_sep_uv(int grid_size, int grid_stride,
                                     int kernel_size, int oversample,
                                     double theta,
                                     double u, double v,
                                     int *grid_offset,
                                     int *sub_offset_x, int *sub_offset_y)
{

    double x = theta * u, y = theta * v;
    // Find fractional coordinates
    int ix, iy, ixf, iyf;
    _frac_coord(grid_size, oversample, x, &ix, &ixf);
    _frac_coord(grid_size, oversample, y, &iy, &iyf);
    // Calculate grid and oversampled kernel offsets
    *grid_offset = (iy-kernel_size/2)*grid_stride + (ix-kernel_size/2);
    *sub_offset_x = kernel_size * ixf;
    *sub_offset_y = kernel_size * iyf;
}

inline static
void degrid_conv_uv(double complex *uvgrid, int grid_size, int grid_stride, double theta,
                    double complex mult,
                    double u, double v,
                    struct sep_kernel_data *kernel, int kernel_size,
                    uint64_t *flops, double complex *pvis)
{

    // Calculate grid and sub-grid coordinates
    int grid_offset, sub_offset_x, sub_offset_y;
    frac_coord_sep_uv(grid_size, grid_stride, kernel_size, kernel->oversampling,
                      theta, u, v,
                      &grid_offset, &sub_offset_x, &sub_offset_y);

#ifndef __AVX2__

    // Get visibility
    double complex vis = 0;
    int y, x;
    for (y = 0; y < kernel_size; y++) {
        double complex visy = 0;
        for (x = 0; x < kernel_size; x++) {
            visy += kernel->data[sub_offset_x + x] *
                    uvgrid[grid_offset + y*grid_stride + x];
        }
        vis += kernel->data[sub_offset_y + y] * visy;
    }
    *flops += 4 * (1 + kernel_size) * kernel_size;
    *pvis = creal(vis) * creal(mult) + I * cimag(vis) * cimag(mult);
#else

    double complex *pgrid = uvgrid + grid_offset;
    double *kernel_x = kernel->data + sub_offset_x;
    double *kernel_y = kernel->data + sub_offset_y;

    // Get visibility
    assert(kernel_size % 2 == 0);
    __m256d vis = _mm256_setzero_pd();
    int y, x;
    for (y = 0; y < kernel_size; y += 1) {
        __m256d grid = _mm256_loadu_pd((double *)(pgrid + y*grid_stride));
        double *pk = kernel_x;
        __m256d kern = _mm256_setr_pd(*pk, *pk, *(pk+1), *(pk+1));
        __m256d sum = _mm256_mul_pd(kern, grid);
        _mm_prefetch(uvgrid + grid_offset + (y+1)*grid_stride, _MM_HINT_T0);
        for (x = 2, pk += 2; x < kernel_size; x += 2, pk += 2) {
            __m256d kern = _mm256_setr_pd(*pk, *pk, *(pk+1), *(pk+1));
            __m256d grid = _mm256_loadu_pd((double *)(pgrid + y*grid_stride + x));
            sum = _mm256_fmadd_pd(kern, grid, sum);
        }
        double kern_y = kernel_y[y];
        vis = _mm256_fmadd_pd(sum, _mm256_set1_pd(kern_y), vis);
    }

    __m128d vis_out = _mm256_extractf128_pd(vis, 0) + _mm256_extractf128_pd(vis, 1);
    _mm_store_pd((double *)pvis, vis_out * _mm_load_pd((double *)&mult));

    *flops += 4 * (1 + kernel_size) * kernel_size;
#endif
}

inline static int imax(int a, int b) { return a >= b ? a : b; }
inline static int imin(int a, int b) { return a <= b ? a : b; }

void degrid_conv_uv_line(double complex *uvgrid, int grid_size, int grid_stride, double theta,
                         double u0, double v0, double du, double dv, int count,
                         double min_u, double max_u, double min_v, double max_v,
                         bool conjugate,
                         struct sep_kernel_data *kernel,
                         double complex *pvis0, uint64_t *flops)
{

    // Figure out bounds
    int i0 = 0, i1 = count;
    if (du > 0) {
        if (u0+i0*du < min_u) i0 = imax(i0, ceil( (min_u - u0) / du ));
        if (u0+i1*du > max_u) i1 = imin(i1, ceil( (max_u - u0) / du ));
    } else if (du < 0) {
        if (u0+i0*du > max_u) i0 = imax(i0, ceil( (max_u - u0) / du ));
        if (u0+i1*du < min_u) i1 = imin(i1, ceil( (min_u - u0) / du ));
    }
    if (dv > 0) {
        if (v0+i0*dv < min_v) i0 = imax(i0, ceil( (min_v - v0) / dv ));
        if (v0+i1*dv > max_v) i1 = imin(i1, ceil( (max_v - v0) / dv ));
    } else if (dv < 0) {
        if (v0+i0*dv > max_v) i0 = imax(i0, ceil( (max_v - v0) / dv ));
        if (v0+i1*dv < min_v) i1 = imin(i1, ceil( (min_v - v0) / dv ));
    }
    i0 = imax(0, imin(i0, i1));

    // Fill zeroes
    int i = 0; double complex *pvis = pvis0;
    double u = u0, v = v0;
    for (; i < i0; i++, u += du, v += dv, pvis++) {
        //assert(!(u >= min_u && u < max_u &&
        //         v >= min_v && v < max_v));
        *pvis = 0.;
    }

    const int kernel_size = kernel->size;

    // Anything to do?
    if (i < i1) {

        {

            double complex mult = (conjugate ? 1 - I : 1 + I);
            for (; i < i1; i++, u += du, v += dv, pvis+=1) {
                assert(u >= min_u && u < max_u &&
                       v >= min_v && v < max_v);
                degrid_conv_uv(uvgrid, grid_size, grid_stride, theta,
                               mult, u, v, kernel, kernel_size, flops, pvis);
            }

        }
    }

    // Fill remaining zeroes
    for (; i < count; i++, u += du, v += dv, pvis++) {
        //assert(!(u >= min_u && u < max_u &&
        //       v >= min_v && v < max_v));
        *pvis = 0.;
    }

}


uint64_t degrid_conv_bl(double complex *uvgrid, int grid_size, int grid_stride, double theta,
                        double d_u, double d_v,
                        double min_u, double max_u, double min_v, double max_v,
                        struct bl_data *bl,
                        int time0, int time1, int freq0, int freq1,
                        struct sep_kernel_data *kernel)
{
    uint64_t flops = 0;
    int time, freq;
    for (time = time0; time < time1; time++) {
        for (freq = freq0; freq < freq1; freq++) {

            // Bounds check
            double u = uvw_lambda(bl, time, freq, 0);
            if (u < min_u || u >= max_u) continue;
            double v = uvw_lambda(bl, time, freq, 1);
            if (v < min_v || v >= max_v) continue;

            degrid_conv_uv(uvgrid, grid_size, grid_stride, theta,
                           1+I,
                           u-d_u, v-d_v, kernel, kernel->size, &flops,
                           bl->vis + (time-time0)*(freq1 - freq0) + freq-freq0);
        }
    }

    return flops;
}

void fft_shift(double complex *uvgrid, int grid_size) {

    // Shift the FFT
    assert(grid_size % 2 == 0);
    int x, y;
    for (y = 0; y < grid_size; y++) {
        for (x = 0; x < grid_size/2; x++) {
            int ix0 = y * grid_size + x;
            int ix1 = (ix0 + (grid_size+1) * (grid_size/2)) % (grid_size*grid_size);
            double complex temp = uvgrid[ix0];
            uvgrid[ix0] = uvgrid[ix1];
            uvgrid[ix1] = temp;
        }
    }

}
