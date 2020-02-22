
#include "grid.h"
#include "config.h"
#include "streamer.h"

#include <hdf5.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <omp.h>
#include <float.h>
#include <pthread.h>

#include <sys/mman.h>
#include <sys/wait.h>

uint64_t streamer_degrid_worker(struct streamer *streamer,
                                struct bl_data *bl_data,
                                int SG_stride, double complex *subgrid,
                                double mid_u, double mid_v, double mid_w,
                                int iu, int iv, int iw,
                                bool conjugate,
                                int it0, int it1, int if0, int if1,
                                double min_u, double max_u,
                                double min_v, double max_v,
                                double min_w, double max_w,
                                double complex *vis_data)
{
    struct vis_spec *const spec = &streamer->work_cfg->spec;
    const double theta = streamer->work_cfg->theta;
    const int subgrid_size = streamer->work_cfg->recombine.xM_size;

    // Calculate l/m positions of sources in case we are meant to
    // check them
    int i;
    const int image_size = streamer->work_cfg->recombine.image_size;
    const int source_count = streamer->work_cfg->source_count;

    // Initialise counter to random value so we check random visibilities
    int source_checks = streamer->work_cfg->vis_checks;
    int check_counter = 0;
    if (source_checks > 0 && source_count > 0) {
        check_counter = rand() % source_checks;
    } else {
        source_checks = 0;
    }

    // Do degridding
    uint64_t flops = 0;
    uint64_t square_error_samples = 0;
    double square_error_sum = 0, worst_err = 0;
    int time;
    for (time = it0; time < it1; time++) {

        // Determine coordinates
        double u = uvw_lambda(bl_data, time, if0, 0);
        double v = uvw_lambda(bl_data, time, if0, 1);
        double w = uvw_lambda(bl_data, time, if0, 2);
        double du = uvw_lambda(bl_data, time, if0+1, 0) - u;
        double dv = uvw_lambda(bl_data, time, if0+1, 1) - v;
        double dw = uvw_lambda(bl_data, time, if0+1, 2) - w;
        if (conjugate) {
            u *= -1; du *= -1; v *= -1; dv *= -1; w *= -1; dw *= -1;
        }

        // Degrid a line of visibilities
        double complex *pvis = vis_data + (time-it0)*spec->freq_chunk;
        //if (fabs(w-mid_w) > 300) {
            //}
        degrid_conv_uv_line(subgrid, subgrid_size, SG_stride, theta,
                            u-mid_u, v-mid_v, w-mid_w, du, dv, dw, if1 - if0,
                            min_u-mid_u, max_u-mid_u,
                            min_v-mid_v, max_v-mid_v,
                            min_w-mid_w, max_w-mid_w,
                            conjugate,
                            streamer->kern, pvis, &flops);

        // Check against DFT (one per row, maximum)
        if (source_checks > 0) {
            if (check_counter >= if1 - if0) {
                check_counter -= if1 - if0;

            } else {

                double complex vis_out = pvis[check_counter];
                double check_u = u + du * check_counter;
                double check_v = v + dv * check_counter;
                double check_w = w + dw * check_counter;

                // Check that we actually generated a visibility here,
                // negate if necessary
                complex double vis = 0;
                if (check_u >= min_u && check_u < max_u &&
                    check_v >= min_v && check_v < max_v &&
                    check_w >= min_w && check_w < max_w) {


                    if (streamer->work_cfg->vis_round_to_wplane) {
                        if (fabs(check_w - mid_w) >
                            streamer->work_cfg->wstep / 2) {

                            printf("check_u=%g mid_u=%g diff=%g\n",
                                   check_u, mid_u, fabs(check_u - mid_u));
                            printf("check_v=%g mid_v=%g diff=%g\n",
                                   check_v, mid_v, fabs(check_v - mid_v));
                            printf("check_w=%g mid_w=%g diff=%g\n",
                                   check_w, mid_w, fabs(check_w - mid_w));
                            assert(fabs(check_w - mid_w) <= streamer->work_cfg->wstep / 2);
                        }
                        check_w = mid_w;
                    }

                    if (conjugate) { check_u *= -1; check_v *= -1; check_w *= -1; }

                    // Generate visibility
                    for (i = 0; i < source_count; i++) {
                        double ph =
                            check_u * streamer->work_cfg->source_lmn[i*3+0] +
                            check_v * streamer->work_cfg->source_lmn[i*3+1] +
                            check_w * streamer->work_cfg->source_lmn[i*3+2];
                        vis += cos(2*M_PI*ph) + 1.j * sin(2*M_PI*ph);
                    }
                    vis /= (double)image_size * image_size;

                }

                // Check error
                square_error_samples += 1;
                double err = cabs(vis_out - vis);
                if (err > 1e-5) {
                    printf("%d  %g/%g/%g-%g/%g/%g (sg %g/%g/%g x %g/%g/%g)\n",
                           time,
                           u, v, w, u+du*(if1-if0),v+dv*(if1-if0),w+dw*(if1-if0),
                           min_u, min_v, min_w,
                           max_u, max_v, max_w);
                    for (i = 0; i < 128; i++) {
                        printf(" %g%+gj ", creal(subgrid[i]), cimag(subgrid[i]));
                    }
                    puts("");
                    for (i = 0; i < if1-if0; i++) {
                        printf(" %g%+gj ", creal(pvis[i]), cimag(pvis[i]));
                    }
                    printf("\nWARNING: uv %g/%g (sg %d/%d): %g%+gj != %g%+gj\n",
                           check_u, check_v, iu, iv, creal(vis_out), cimag(vis_out), creal(vis), cimag(vis));
                }
                worst_err = fmax(err, worst_err);
                square_error_sum += err * err;

                check_counter = source_checks;
            }
        }
    }

    // Add to statistics
    #pragma omp atomic
        streamer->vis_error_samples += square_error_samples;
    #pragma omp atomic
        streamer->vis_error_sum += square_error_sum;
    if (worst_err > streamer->vis_worst_error) {
        // Likely happens rarely enough that a critical section won't be a problem
        #pragma omp critical
           streamer->vis_worst_error = fmax(streamer->vis_worst_error, worst_err);
    }
    #pragma omp atomic
        streamer->degrid_flops += flops;
    #pragma omp atomic
        streamer->produced_chunks += 1;

    return flops;
}

bool streamer_degrid_chunk(struct streamer *streamer,
                           struct subgrid_work *work,
                           struct subgrid_work_bl *bl,
                           int tchunk, int fchunk,
                           int slot,
                           int SG_stride, double complex *subgrid)
{
    struct vis_spec *const spec = &streamer->work_cfg->spec;
    struct recombine2d_config *const cfg = &streamer->work_cfg->recombine;
    const double theta = streamer->work_cfg->theta;
    const double wstep = streamer->work_cfg->wstep;
    const double sg_step = streamer->work_cfg->sg_step;
    const double sg_step_w = streamer->work_cfg->sg_step_w;

    double start = get_time_ns();

    // Calculate subgrid boundaries. TODO: All of this duplicates
    // logic that also appears in config.c (bin_baseline). This is
    // brittle, should get refactored at some point!
    double sg_mid_u = work->subgrid_off_u / theta;
    double sg_mid_v = work->subgrid_off_v / theta;
    double sg_mid_w = work->subgrid_off_w * wstep;
    double sg_min_u = (work->subgrid_off_u - sg_step / 2) / theta;
    double sg_min_v = (work->subgrid_off_v - sg_step / 2) / theta;
    double sg_min_w = (work->subgrid_off_w - sg_step_w / 2) * wstep;
    double sg_max_u = (work->subgrid_off_u + sg_step / 2) / theta;
    double sg_max_v = (work->subgrid_off_v + sg_step / 2) / theta;
    double sg_max_w = (work->subgrid_off_w + sg_step_w / 2) * wstep;
    if (sg_min_v > cfg->image_size / theta / 2) {
        sg_min_v -= cfg->image_size / theta / 2;
        sg_max_v -= cfg->image_size / theta / 2;
    }

    // Determine chunk size
    int it0 = tchunk * spec->time_chunk,
        it1 = (tchunk+1) * spec->time_chunk;
    if (it1 > spec->time_count) it1 = spec->time_count;
    int if0 = fchunk * spec->freq_chunk,
        if1 = (fchunk+1) * spec->freq_chunk;
    if (if1 > spec->freq_count) if1 = spec->freq_count;

    // Check whether time chunk fall into positive u. We use this
    // for deciding whether coordinates are going to get flipped
    // for the entire chunk. This is assuming that a chunk is
    // never big enough that we would overlap an extra subgrid
    // into the negative direction.
    int tstep_mid = (it0 + it1) / 2;
    bool positive_u = bl->bl_data->uvw_m[tstep_mid * 3] >= 0;

    // Check for overlap between baseline chunk and subgrid
    double min_uvw[3], max_uvw[3];
    bl_bounding_box(bl->bl_data, !positive_u, it0, it1-1, if0, if1-1,
                    min_uvw, max_uvw);
    double overlap =
        -fmax((fmax(min_uvw[0], sg_min_u) - fmin(max_uvw[0], sg_max_u)),
              (fmax(min_uvw[1], sg_min_v) - fmin(max_uvw[1], sg_max_v)));
    if (!(min_uvw[0] < sg_max_u && max_uvw[0] > sg_min_u &&
          min_uvw[1] < sg_max_v && max_uvw[1] > sg_min_v &&
          min_uvw[2] < sg_max_w && max_uvw[2] > sg_min_w))
        return false;
    /* printf("%d-%d it=%d-%d if=%d-%d " */
    /*        "u%g-%g v%g-%g w%g-%g " */
    /*        "overlap=%g\n", */
    /*        bl->bl_data->antenna1, bl->bl_data->antenna2, */
    /*        it0,it1,if0,if1, */
    /*        min_uvw[0], max_uvw[0], //sg_min_u, sg_max_u, */
    /*        min_uvw[1], max_uvw[1], //sg_min_v, sg_max_v, */
    /*        min_uvw[2], max_uvw[2], //sg_min_w, sg_max_w, */
    /*        overlap); */

    // Determine least busy writer
    int i, least_waiting = 2 * streamer->vis_queue_per_writer;
    struct streamer_writer *writer = streamer->writer;
    for (i = 0; i < streamer->writer_count; i++) {
        if (streamer->writer[i].to_write < least_waiting) {
            least_waiting = streamer->writer[i].to_write;
            writer = streamer->writer + i;
        }
    }

    // Acquire a slot
    struct streamer_chunk *chunk
        = writer_push_slot(writer, bl->bl_data, tchunk, fchunk);
    #pragma omp atomic
        streamer->wait_in_time += get_time_ns() - start;
    start = get_time_ns();

    // Do degridding
    const size_t chunk_vis_size = sizeof(double complex) * spec->freq_chunk * spec->time_chunk;
    uint64_t flops = streamer_degrid_worker(
        streamer, bl->bl_data, SG_stride, subgrid,
        sg_mid_u, sg_mid_v, sg_mid_w,
        work->iu, work->iv, work->iw,
        !positive_u,
        it0, it1, if0, if1,
        sg_min_u, sg_max_u, sg_min_v, sg_max_v, sg_min_w, sg_max_w,
        chunk ? chunk->vis : alloca(chunk_vis_size));
    #pragma omp atomic
      streamer->degrid_time += get_time_ns() - start;

      //assert(flops > 0 || overlap < 80);
    if (chunk) {

        // No flops executed? Signal to writer that we can skip writing
        // this chunk (small optimisation)
        if (flops == 0) {
            chunk->tchunk = -2;
            chunk->fchunk = -2;
        }

        // Signal slot for output
#ifndef __APPLE__
        sem_post(&chunk->out_lock);
#else
        dispatch_semaphore_signal(chunk->out_lock);
#endif
    }

    return true;
}

// How is this not in the standard library somewhere?
// (stolen from Stack Overflow)
inline static double complex cipow(double complex base, int exp)
{
    double complex result = 1;
    if (exp < 0) return 1 / cipow(base, -exp);
    if (exp == 1) return base;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}

void streamer_task(struct streamer *streamer,
                   struct subgrid_work *work,
                   struct subgrid_work_bl *bl_start,
                   int slot,
                   int subgrid_work,
                   double complex *subgrid_image)
{


    // Determine subgrid dimensions.
    const int xM_size = streamer->work_cfg->recombine.xM_size;
    const int SG_stride = xM_size + 16; // Assume 16x16 is biggest possible convolution
    const int SG2_size = sizeof(double complex) * SG_stride * xM_size;
    double complex *subgrid = calloc(1, SG2_size);

    // Determine how much space we need to cover along the w-axis
    struct subgrid_work_bl *bl;
    int i_bl; double min_w = bl_start->min_w, max_w = bl_start->max_w;
    for (bl = bl_start->next, i_bl = 1;
         bl && i_bl < streamer->work_cfg->vis_bls_per_task;
         bl = bl->next, i_bl++) {

        min_w = fmin(min_w, bl->min_w);
        max_w = fmax(max_w, bl->max_w);
    }

    // Determine w-planes need to cover
    const double wstep = streamer->work_cfg->wstep;
    int w_start = (int) floor(min_w / wstep + 0.5) - work->subgrid_off_w;
    int w_end = (int) floor(max_w / wstep + 0.5) - work->subgrid_off_w;

    //printf("w_min = %g (%d), w_max = %g (%d)\n", min_w, w_start, max_w, w_end);

    int wplane;
    for (wplane = w_start; wplane <= w_end; wplane++) {

        // FFT and establish proper stride for the subgrid so we don't get
        // cache thrashing problems when gridding moves (TODO: construct
        // like this right away? Shouldn't FFTW be able to do this?)
        if (subgrid_image) {
            double start_time = get_time_ns();

            // Apply w-transfer pattern
            int i;
            double complex *const wtransfer = streamer->wtransfer;
            printf("wplane=%d (min_wp=%g)\n", wplane,
                   min_w / wstep - work->subgrid_off_w);
            for (i = 0; i < xM_size * xM_size; i++) {
                subgrid[i] = subgrid_image[i] * cipow(wtransfer[i], wplane);
            }
            fftw_execute_dft(streamer->subgrid_plan, subgrid, subgrid);
            fft_shift(subgrid, xM_size);
            for (i = xM_size-1; i >= 0; i--) {
                memmove(subgrid + SG_stride * i,
                        subgrid + xM_size * i,
                        sizeof(double complex) * xM_size);
            }
            streamer->fft_time += get_time_ns() - start_time;
        }

        struct vis_spec *const spec = &streamer->work_cfg->spec;
        for (bl = bl_start, i_bl = 0;
             bl && i_bl < streamer->work_cfg->vis_bls_per_task;
             bl = bl->next, i_bl++) {

            // Go through time/frequency chunks
            int ntchunk = (bl->bl_data->time_count + spec->time_chunk - 1) / spec->time_chunk;
            int nfchunk = (bl->bl_data->freq_count + spec->freq_chunk - 1) / spec->freq_chunk;
            int tchunk, fchunk;
            int nchunks = 0;
            for (tchunk = 0; tchunk < ntchunk; tchunk++)
                for (fchunk = 0; fchunk < nfchunk; fchunk++)
                    if (streamer_degrid_chunk(streamer, work,
                                              bl, tchunk, fchunk,
                                              slot, SG_stride, subgrid))
                        nchunks++;

            // Check that plan predicted the right number of chunks. This
            // is pretty important - if this fails this means that the
            // coordinate calculations are out of synch, which might mean
            // that we have failed to account for some visibilities in the
            // plan!
            if (bl->chunks != nchunks)
                printf("WARNING: subgrid (%d/%d/%d) baseline (%d-%d) %d chunks planned, %d actual!\n",
                       work->iu, work->iv, work->iw, bl->a1, bl->a2, bl->chunks, nchunks);

        }

    }

    // Done with this chunk
#pragma omp atomic
    streamer->subgrid_tasks--;
#pragma omp atomic
    streamer->subgrid_locks[slot]--;

    free(subgrid);
}

// Perform checks on the subgrid data. Returns RMSE if we have sources
// to do the checks.
static double streamer_checks(struct streamer *streamer,
                              struct subgrid_work *work,
                              complex double *subgrid_image)
{
    struct recombine2d_config *const cfg = &streamer->work_cfg->recombine;

    // Perform Fourier transform
    complex double *subgrid = calloc(sizeof(complex double), cfg->SG_size);
    fftw_execute_dft(streamer->subgrid_plan, subgrid_image, subgrid);

    // Check accumulated result
    if (work->check_path && streamer->work_cfg->facet_workers > 0) {
        double complex *approx_ref = read_hdf5(cfg->SG_size, work->check_hdf5, work->check_path);
        double err_sum = 0; int y;
        for (y = 0; y < cfg->xM_size * cfg->xM_size; y++) {
            double err = cabs(subgrid[y] - approx_ref[y]); err_sum += err * err;
        }
        free(approx_ref);
        double rmse = sqrt(err_sum / cfg->xM_size / cfg->xM_size);
        printf("%sSubgrid %d/%d RMSE %g\n",
               rmse > work->check_threshold ? "ERROR: " : "",
               work->iu, work->iv, rmse);

    }

    fft_shift(subgrid, cfg->xM_size);

    // Check some degridded example visibilities
    if (work->check_degrid_path && streamer->kern && streamer->work_cfg->facet_workers > 0) {
        int nvis = get_npoints_hdf5(work->check_hdf5, "%s/vis", work->check_degrid_path);
        double *uvw_sg = read_hdf5(3 * sizeof(double) * nvis, work->check_hdf5,
                                   "%s/uvw_subgrid", work->check_degrid_path);
        int vis_size = sizeof(double complex) * nvis;
        double complex *vis = read_hdf5(vis_size, work->check_hdf5,
                                        "%s/vis", work->check_degrid_path);

        struct bl_data bl;
        bl.antenna1 = bl.antenna2 = 0;
        bl.time_count = nvis;
        bl.freq_count = 1;
        double freq[] = { c }; // 1 m wavelength
        bl.freq = freq;
        bl.vis = (double complex *)calloc(1, vis_size);

        // Degrid and compare
        bl.uvw_m = uvw_sg;
        degrid_conv_bl(subgrid, cfg->xM_size, cfg->xM_size, cfg->image_size, 0, 0,
                       -cfg->xM_size, cfg->xM_size, -cfg->xM_size, cfg->xM_size,
                       &bl, 0, nvis, 0, 1, streamer->kern);
        double err_sum = 0; int y;
        for (y = 0; y < nvis; y++) {
            double err = cabs(vis[y] - bl.vis[y]); err_sum += err*err;
        }
        double rmse = sqrt(err_sum / nvis);
        printf("%sSubgrid %d/%d degrid RMSE %g\n",
               rmse > work->check_degrid_threshold ? "ERROR: " : "",
               work->iu, work->iv, rmse);

    }

    // Check against DFT, if we are generating from sources
    const int source_count = streamer->work_cfg->source_count;
    double err_sum = 0, worst_err = 0; int err_samples = 0;
    int source_checks = streamer->work_cfg->grid_checks;
    if (source_count > 0 && source_checks > 0) {

        const double theta = streamer->work_cfg->theta;
        const double wstep = streamer->work_cfg->wstep;

        int iu, iv;
        int check_counter = rand() % source_checks;
        for (iv = -cfg->xA_size/2; iv < cfg->xA_size/2; iv++) {
            for (iu = -cfg->xA_size/2; iu < cfg->xA_size/2; iu++) {
                if (check_counter--) continue;
                check_counter = source_checks;

                double check_u = (work->subgrid_off_u+iu) / theta;
                double check_v = (work->subgrid_off_v+iv) / theta;
                double check_w = work->subgrid_off_w * wstep;

                // Generate visibility
                complex double vis = 0;
                int i;
                for (i = 0; i < source_count; i++) {
                    double ph =
                        check_u * streamer->work_cfg->source_lmn[i*3+0] +
                        check_v * streamer->work_cfg->source_lmn[i*3+1] +
                        check_w * streamer->work_cfg->source_lmn[i*3+2];
                    vis += 1/streamer->work_cfg->source_corr[i]
                        * (cos(2*M_PI*ph) + 1.j * sin(2*M_PI*ph));
                }
                vis /= (double)cfg->image_size * cfg->image_size;

                // Check
                double complex  vis_grid = subgrid[(iv+cfg->xM_size/2) * cfg->xM_size + iu + cfg->xM_size/2];
                double err = cabs(vis_grid - vis);
                err_sum += err * err;
                worst_err = fmax(worst_err, err);
                err_samples += 1;

            }

        }

        #pragma omp atomic
             streamer->grid_error_samples += err_samples;
        #pragma omp atomic
             streamer->grid_error_sum += err_sum;
        if (worst_err > streamer->grid_worst_error) {
            #pragma omp critical
                 streamer->grid_worst_error = fmax(streamer->grid_worst_error, worst_err);
        }

    }

    free(subgrid);
    if (err_samples > 0)
        return sqrt(err_sum / err_samples) / streamer->work_cfg->source_energy;
    else
        return -1;
}

void streamer_work(struct streamer *streamer,
                   int subgrid_work,
                   double complex *nmbf)
{

    struct recombine2d_config *const cfg = &streamer->work_cfg->recombine;
    struct subgrid_work *const work = streamer->work_cfg->subgrid_work +
        streamer->subgrid_worker * streamer->work_cfg->subgrid_max_work + subgrid_work;
    struct facet_work *const facet_work = streamer->work_cfg->facet_work;

    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    const int nmbf_length = cfg->NMBF_NMBF_size / sizeof(double complex);

    // Find slot to write to
    int slot;
    while(true) {
        for (slot = 0; slot < streamer->queue_length; slot++)
            if (streamer->subgrid_locks[slot] == 0)
                break;
        if (slot < streamer->queue_length)
            break;
        #pragma omp taskyield
        usleep(100);
    }

    double recombine_start = get_time_ns();

    // Compare with reference
    if (work->check_fct_path) {

        int i0 = work->iv, i1 = work->iu;
        int ifacet;
        for (ifacet = 0; ifacet < facets; ifacet++) {
            if (!facet_work[ifacet].set) continue;
            int j0 = facet_work[ifacet].im, j1 = facet_work[ifacet].il;
            double complex *ref = read_hdf5(cfg->NMBF_NMBF_size, work->check_hdf5,
                                            work->check_fct_path, j0, j1);
            int x; double err_sum = 0;
            for (x = 0; x < nmbf_length; x++) {
                double err = cabs(ref[x] - nmbf[nmbf_length*ifacet+x]); err_sum += err*err;
            }
            free(ref);
            double rmse = sqrt(err_sum / nmbf_length);
            if (!work->check_fct_threshold || rmse > work->check_fct_threshold) {
                printf("Subgrid %d/%d facet %d/%d checked: %g RMSE\n",
                       i0, i1, j0, j1, rmse);
            }
        }
    }

    // Accumulate contributions to this subgrid
    double complex *subgrid = subgrid_slot(streamer, slot);
    memset(subgrid, 0, cfg->SG_size);
    int ifacet;
    for (ifacet = 0; ifacet < facets; ifacet++)
        recombine2d_af0_af1(cfg, subgrid,
                            facet_work[ifacet].facet_off_m,
                            facet_work[ifacet].facet_off_l,
                            nmbf + nmbf_length*ifacet);
    streamer->recombine_time += get_time_ns() - recombine_start;

    // Perform checks on result
    double check_start = get_time_ns();
    double rmse = streamer_checks(streamer, work, subgrid);
    streamer->check_time += get_time_ns() - check_start;

    struct vis_spec *const spec = &streamer->work_cfg->spec;
    if (spec->time_count > 0 && streamer->kern) {

        // Loop through baselines
        struct subgrid_work_bl *bl;
        int i_bl = 0;
        for (bl = work->bls; bl; bl = bl->next, i_bl++) {
            if (i_bl % streamer->work_cfg->vis_bls_per_task != 0)
                continue;

            // We are spawning a task: Add lock to subgrid data to
            // make sure it doesn't get overwritten
            #pragma omp atomic
              streamer->subgrid_tasks++;
            #pragma omp atomic
              streamer->subgrid_locks[slot]++;

            // Start task. Make absolutely sure it sees *everything*
            // as private, as Intel's C compiler otherwise loves to
            // generate segfaulting code. OpenMP complains here that
            // having a "private" constant is unecessary (requiring
            // the copy), but I don't trust its judgement.
            double task_start = get_time_ns();
            struct subgrid_work *_work = work;
            #pragma omp task firstprivate(streamer, _work, bl, slot, subgrid_work, subgrid)
                streamer_task(streamer, _work, bl, slot, subgrid_work, subgrid);
            #pragma omp atomic
                streamer->task_start_time += get_time_ns() - task_start;
        }

        if (rmse >= 0) {
            printf("Subgrid %d/%d/%d (%d baselines, rmse %.02g)\n",
                   work->iu, work->iv, work->iw, i_bl, rmse);
        } else {
            printf("Subgrid %d/%d/%d (%d baselines)\n",
                   work->iu, work->iv, work->iw, i_bl);
        }
        fflush(stdout);
        streamer->baselines_covered += i_bl;

    }
}
