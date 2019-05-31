

import sys
import os
import re

if len(sys.argv) != 2:
    print("Please supply an outpuf file name!", file=sys.stderr)
    exit(1)

# Extract desired kernel size from file name
out_fname = sys.argv[1]
kernel_size = int(re.search('([0-9]+)\.', os.path.basename(out_fname))[1])


# Architecture assumptions (bytes)
double_size = 8
double_complex_size = 16
register_size = 32
cache_line_size = 64
align = 16

# Number of parallel sums. Should be at least the number of parallel
# FMA units present on the CPU core
parallel_sums = 2

# No overflow for the moment
assert (kernel_size * double_size % register_size == 0)

kernel_regs = kernel_size * double_size // register_size

kernel_in_size = kernel_size * double_size
kernel_in_cache_lines = (kernel_in_size + cache_line_size - 1) // cache_line_size
kernel_out_size = kernel_size * double_complex_size


# Open the file
out_file = open(out_fname, "w")
def emit(*args, **kwargs):
    print(*args, **kwargs, file=out_file)

# Generate prelude. We assume that conjugation can be achieved using
# an "xor" operation (we essentially just flip a sign).
emit(f"""

// THIS IS A GENERATED FILE!

assert(kernel->size == {kernel_size});
assert((uintptr_t)kernel->data % {align} == 0);
assert((uintptr_t)pvis % {align} == 0);

const int oversample = kernel->oversampling;
__m128d conj_mask = conjugate ? _mm_xor_pd(_mm_set_pd(-1, 1), _mm_set_pd(1, 1)) : _mm_set_pd(0., 0.);

// Calculate grid and sub-grid coordinates
int grid_offset, sub_offset_x, sub_offset_y;
frac_coord_sep_uv(grid_size, grid_stride, {kernel_size}, oversample,
                  theta, u, v,
                  &grid_offset, &sub_offset_x, &sub_offset_y);
""")

# Load first Y kernel
kernel_y_load = True
unroll_y_loop = True
if kernel_y_load:

    if not unroll_y_loop:
        emit(f"__m256d kern_y_cache[{kernel_size * double_size // register_size}];")

    def load_kernel(ind, in_ptr, out_vars, declare=False):
        """Load kernels into variables (registers, possibly)"""

        dec = ("__m256d " if declare else "")
        for s in range(kernel_regs):
            off = s * register_size // double_size
            emit(f"{ind}{dec}{out_vars[s]} = _mm256_load_pd({in_ptr}+{off});")

    kern_y_vars = [ f"kern_y_cache[{s}]" if not unroll_y_loop else f"kern_y_{s}"
                    for s in range(2 * kernel_regs) ]
    load_kernel("", "kernel->data+sub_offset_y", kern_y_vars, unroll_y_loop)

else:

    emit(f"__attribute__((aligned({align}))) double kern_y_cache[{kernel_size}];")

    def copy_kernel(ind, in_ptr, out_ptr):
        """ Copy kernel from and to memory """
        for s in range(kernel_regs):
            off = s * register_size // double_size
            emit(f"{ind}_mm256_store_pd({out_ptr}+{off}, _mm256_load_pd({in_ptr}+{off}));")
    copy_kernel("", "kernel->data+sub_offset_y", "kern_y_cache")

# Load first X kernel
def make_permute(a,b,c,d):
    assert(all([x >= 0 for x in [a,b,c,d]]))
    assert(all([x < register_size // double_size for x in [a,b,c,d]]))
    return (a << 0) | (b << 2) | (c << 4) | (d << 6)

def load_kernel_dup(ind, in_ptr, out_vars, declare=False):
    """Load kernels into variables (registers, hopefully), duplicating all
    values in the process."""

    dec = ("__m256d " if declare else "")
    for s in range(kernel_regs):
        off = s * register_size // double_size
        permut1 = make_permute(0,0,1,1)
        permut2 = make_permute(2,2,3,3)
        emit(f"{ind}__m256d k{s} = _mm256_load_pd({in_ptr}+{off});")
        emit(f"{ind}{dec}{out_vars[2*s]} = _mm256_permute4x64_pd(k{s},{permut1});")
        emit(f"{ind}{dec}{out_vars[2*s+1]} = _mm256_permute4x64_pd(k{s},{permut2});")

kern_x_vars = [ f"kern_x_{s}" for s in range(2 * kernel_regs) ]
load_kernel_dup("", "kernel->data+sub_offset_x", kern_x_vars, declare=True)

# Now generate loop over visibilities
emit(f"""
complex double *next_grid = uvgrid + grid_offset;
for (; i < i1; i++, u += du, v += dv, pvis++) {{

    int grid_offset, sub_offset_x, sub_offset_y;
    frac_coord_sep_uv(grid_size, grid_stride, {kernel_size}, oversample,
                      theta, u+du, v+dv,
                      &grid_offset, &sub_offset_x, &sub_offset_y);

    complex double *current_grid = next_grid;
    next_grid = uvgrid + grid_offset;

    double *next_kernel_x = kernel->data + sub_offset_x;
    double *next_kernel_y = kernel->data + sub_offset_y;""")

# Prefetch kernel for next visibility
def prefetch_kernel(ind, ptr):
    for s in range(kernel_size * double_size // cache_line_size):
        off = s * cache_line_size // double_size
        emit(f"{ind}_mm_prefetch({ptr}+{off}, _MM_HINT_T0);")
prefetch_kernel("    ", "next_kernel_x")
prefetch_kernel("    ", "next_kernel_y")

def sum_row(ind, grid_ptr, kern_vars, suff=""):

    # Determine how many terms we need to sum, and how many parallel
    # sums we are going to keep
    terms = kernel_size * double_complex_size // register_size
    sum_vars = [ f"sum{suff}{s}" for s in range(min(terms, parallel_sums)) ]
    flops = 0
    for s, (svar, kvar) in enumerate(zip(sum_vars, kern_vars)):
        off = s * register_size // double_size
        emit(f"{ind}__m256d {svar} = _mm256_mul_pd({kvar}, _mm256_loadu_pd({grid_ptr}+{off}));")
        flops += register_size // double_size
    # Cover remaining row
    for o in range(parallel_sums, terms, parallel_sums):
        for s, (svar, kvar) in enumerate(zip(sum_vars, kern_vars[o:])):
            off = (s+o) * register_size // double_size
            emit(f"{ind}{svar} = _mm256_fmadd_pd({kvar}, _mm256_loadu_pd({grid_ptr}+{off}), {svar});");
            flops += 2 * register_size // double_size
    # Generate sum term to return
    flops += (len(sum_vars) - 1) * register_size // double_size
    return "+".join(sum_vars), flops

kernel_y_reg_cache = False
def sum_row_group(ind, grid_ptr, y_expr, suff=""):

    if kernel_y_load:
        if unroll_y_loop:
            kern_y_expr = kern_y_vars[int(y_expr)*double_size//register_size]
        else:
            kern_y_expr = f"kern_y_cache[{y_expr}/{register_size//double_size}]"
    else:
        kern_y_expr = f"_mm256_load_pd(kern_y_cache + {y_expr})"

    if kernel_y_reg_cache:
        emit(f"{ind}__m256d kerny{suff} = {kern_y_expr};")
        kern_y_expr = f"kerny{suff}"

    flops_sum = 0
    for u in range(register_size // double_size):
        # Get pointer to grid row
        emit(f"{ind}double *pgrid{suff}{u} = (double *)(current_grid + ({y_expr}+{u})*grid_stride);")
        # Permute the y-kernel appropriately. We do this before loading
        # the grid line, as the permutation has quite a bit of latency
        permut = make_permute(u,u,u,u)
        emit(f"{ind}__m256d ky{suff}{u} = _mm256_permute4x64_pd({kern_y_expr},{permut});")
        # Generate sum over row, multipy/add into visibility accumulator
        sum_term, flops = sum_row(ind, f"pgrid{suff}{u}", kern_x_vars, suff=suff+str(u))
        emit(f"{ind}vis = _mm256_fmadd_pd({sum_term}, ky{suff}{u}, vis);")
        flops_sum += flops + 2 * register_size // double_size

    return flops_sum

# Now loop over rows. This can be done either as a loop over
# groups, or unrolled entirely
if unroll_y_loop:
    flops_sum = 0
    emit(f"""    __m256d vis = _mm256_setzero_pd();""")
    for y in range(0, kernel_size, register_size // double_size):
        flops_sum += sum_row_group("    ", "current_grid", str(y), suff=str(y))

else:
    # Generate inner loop, unrolled so we can load as many values of the
    # y-direction kernel at once as possible
    emit(f"""
    __m256d vis = _mm256_setzero_pd();
    int y;
    for (y = 0; y < {kernel_size}; y += {register_size // double_size}) {{""")

    flops_sum = sum_row_group("        ", "current_grid", "y")

    emit(f"    }}")
    flops_sum *= register_size // double_size

# Copy next kernel
load_kernel_dup("    ", "next_kernel_x", kern_x_vars)
if kernel_y_load:
    load_kernel("    ", "next_kernel_y", kern_y_vars)
else:
    copy_kernel("    ", "next_kernel_y", "kern_y_cache")

# Store visibility and conjugate. We don't count this floating point
# operations, it's overhead strictly speaking.
emit(f"""    __m128d vis_out = _mm256_extractf128_pd(vis, 0) + _mm256_extractf128_pd(vis, 1);
    _mm_store_pd((double *)pvis, _mm_xor_pd(vis_out, conj_mask));
}}
*flops += {flops_sum} * (i1 - i0);""")

