
#include "grid.h"
#include "config.h"

#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#ifndef NO_MPI
#include <mpi.h>
#else
#define MPI_Request int
#define MPI_REQUEST_NULL 0
#endif

struct producer_stream {

    // Facet worker id, number of facets to work on
    int facet_worker;
    int facet_work_count;

    // Stream targets
    int streamer_count;
    int *streamer_ranks;

    // Send queue
    int send_queue_length;
    MPI_Request *requests;
    uint64_t bytes_sent;

    // Private buffers
    double complex *NMBF_NMBF_queue;

    // Worker structure
    struct recombine2d_worker worker;

    // Time (in s) spent in different stages
    double mpi_wait_time, mpi_send_time;

};

void init_producer_stream(struct recombine2d_config *cfg, struct producer_stream *prod,
                          int facet_worker, int facet_work_count,
                          int streamer_count, int *streamer_ranks,
                          int BF_batch, fftw_plan BF_plan,
                          int send_queue_length)
{

    prod->facet_worker = facet_worker;
    prod->facet_work_count = facet_work_count;

    // Set streamers
    prod->streamer_count = streamer_count;
    prod->streamer_ranks = streamer_ranks;

    // Initialise queue
    prod->send_queue_length = send_queue_length;
    prod->requests = (MPI_Request *) malloc(sizeof(MPI_Request) * send_queue_length);
    int i;
    for (i = 0; i < send_queue_length; i++) {
        prod->requests[i] = MPI_REQUEST_NULL;
    }

    // Create buffers, initialise worker
    prod->NMBF_NMBF_queue =
        (double complex *)malloc(cfg->NMBF_NMBF_size * send_queue_length);
    recombine2d_init_worker(&prod->worker, cfg, BF_batch, BF_plan, FFTW_MEASURE);

    // Initialise statistics
    prod->bytes_sent = 0;
    prod->mpi_wait_time = prod->mpi_send_time = 0;

}

void free_producer_stream(struct producer_stream *prod)
{
    recombine2d_free_worker(&prod->worker);

    free(prod->requests);
    free(prod->NMBF_NMBF_queue);
}

void producer_add_stats(struct producer_stream *to, struct producer_stream *from)
{
    to->bytes_sent += from->bytes_sent;
    to->worker.pf1_time += from->worker.pf1_time;
    to->worker.es1_time += from->worker.es1_time;
    to->worker.ft1_time += from->worker.ft1_time;
    to->worker.pf2_time += from->worker.pf2_time;
    to->worker.es2_time += from->worker.es2_time;
    to->worker.ft2_time += from->worker.ft2_time;
    to->mpi_wait_time += from->mpi_wait_time;
    to->mpi_send_time += from->mpi_send_time;
}

void producer_dump_stats(struct work_config *wcfg, int facet_worker,
                         struct producer_stream *prod,
                         int producer_count, double dt)
{
    struct recombine2d_config *cfg = &wcfg->recombine;

    // For the "effective" statistic we count the number of bytes we
    // conveyed information about. This statistic is slightly messy
    // because on one hand we have communication overheads (decreasing
    // effectiveness), but on the other hand for generating
    // visibilities do not need to cover the entire grid (increasing
    // effectivenes).
    uint64_t effective = 0;
    int i;
    for (i = 0; i < wcfg->facet_max_work; i++) {
        if (wcfg->facet_work[i].set) {
            effective += cfg->F_size;
        }
    }

    double total = dt * producer_count;
    printf("\n%.2f s wall-clock, %.2f GB (%.2f GB effective), %.2f MB/s (%.2f MB/s effective)\n", dt,
           (double)prod->bytes_sent / 1000000000, (double)effective / 1000000000,
           (double)prod->bytes_sent / dt / 1000000, (double)effective / dt/ 1000000);
    printf("PF1: %.2f s (%.1f%%), FT1: %.2f s (%.1f%%), ES1: %.2f s (%.1f%%)\n",
           prod->worker.pf1_time, prod->worker.pf1_time / total * 100,
           prod->worker.ft1_time, prod->worker.ft1_time / total * 100,
           prod->worker.es1_time, prod->worker.es1_time / total * 100);
    printf("PF2: %.2f s (%.1f%%), FT2: %.2f s (%.1f%%), ES2: %.2f s (%.1f%%)\n",
           prod->worker.pf2_time, prod->worker.pf2_time / total * 100,
           prod->worker.ft2_time, prod->worker.ft2_time / total * 100,
           prod->worker.es2_time, prod->worker.es2_time / total * 100);
    double idle = total -
        prod->worker.pf1_time - prod->worker.ft1_time - prod->worker.es1_time -
        prod->worker.pf2_time - prod->worker.ft2_time - prod->worker.es2_time -
        prod->mpi_wait_time - prod->mpi_send_time;
    printf("mpi wait: %.2f s (%.1f%%), mpi send: %.2f s (%.1f%%), idle: %.2f s (%.1f%%)\n",
           prod->mpi_wait_time, 100 * prod->mpi_wait_time / producer_count / dt,
           prod->mpi_send_time, 100 * prod->mpi_send_time / producer_count / dt,
           idle, 100 * idle / producer_count / dt);
}

int make_subgrid_tag(struct work_config *wcfg,
                     int subgrid_worker_ix, int subgrid_work_ix,
                     int facet_worker_ix, int facet_work_ix) {
    // Need to encode only the work items, as with MPI both the sender
    // and the receiver will be identified already by the message.
    return facet_work_ix + subgrid_work_ix * wcfg->facet_max_work;
}

void producer_send_subgrid(struct work_config *wcfg, struct producer_stream *prod,
                           int facet_work_ix,
                           double complex *NMBF_BF,
                           int subgrid_off_u, int subgrid_off_v,
                           int iu, int iv, int iw)
{
    struct recombine2d_config *cfg = &wcfg->recombine;

    // Extract subgrids along second axis
    double complex *NMBF_NMBF = NULL;

    // Find streamer (subgrid workers) to send to
    int iworker;
    for (iworker = 0; iworker < wcfg->subgrid_workers; iworker++) {

        // Check whether it is in streamer's work list. Note that
        // it can appear for multiple workers if the subgrid was
        // split in work assignment (typically at the grid centre).
        struct subgrid_work *work_list = wcfg->subgrid_work +
            iworker * wcfg->subgrid_max_work;
        int iwork;
        for (iwork = 0; iwork < wcfg->subgrid_max_work; iwork++) {
            if (work_list[iwork].nbl &&
                work_list[iwork].iu == iu &&
                work_list[iwork].iv == iv &&
                work_list[iwork].iw == iw)
                break;
        }
        if (iwork >= wcfg->subgrid_max_work)
            continue;

        // Select send slot if running in distributed mode
        int indx;
#ifndef NO_MPI
        if (prod->streamer_count == 0)
            indx = 0;
        else {
            for (indx = 0; indx < prod->send_queue_length; indx++) {
                if (prod->requests[indx] == MPI_REQUEST_NULL) break;
            }
            if (indx >= prod->send_queue_length) {
                double start = get_time_ns();
                MPI_Status status;
                MPI_Waitany(prod->send_queue_length, prod->requests, &indx, &status);
                prod->mpi_wait_time += get_time_ns() - start;
            }
            assert (indx >= 0 && indx < prod->send_queue_length);
        }
#else
        indx = 0;
#endif

        // Calculate or copy sub-grid data
        double complex *send_buf = prod->NMBF_NMBF_queue + indx * cfg->xM_yN_size * cfg->xM_yN_size;
        if (!NMBF_NMBF) {
            NMBF_NMBF = send_buf;
            recombine2d_es0(&prod->worker, subgrid_off_v, subgrid_off_u, NMBF_BF, NMBF_NMBF);
        } else {
            memcpy(send_buf, NMBF_NMBF, cfg->NMBF_NMBF_size);
        }

        // Send (unless running in single-node mode, then we just pretend)
#ifndef NO_MPI
        if (prod->streamer_ranks) {
            int tag = make_subgrid_tag(wcfg, iworker, iwork,
                                       prod->facet_worker, facet_work_ix);
            //printf("Sending iu=%d iv=%d iw=%d tag=%d facet=%d\n",
            //       iu, iv, iw, tag, facet_work_ix);
            double start = get_time_ns();
            MPI_Isend(send_buf, cfg->xM_yN_size * cfg->xM_yN_size, MPI_DOUBLE_COMPLEX,
                      prod->streamer_ranks[iworker], tag, MPI_COMM_WORLD, &prod->requests[indx]);
            prod->mpi_send_time += get_time_ns() - start;
        }
#endif
        prod->bytes_sent += sizeof(double complex) * cfg->xM_yN_size * cfg->xM_yN_size;

    }

}


bool producer_fill_facet(struct work_config *wcfg,
                         struct facet_work *work,
                         double complex *F,
                         int source_count, double *source_xy, double *source_lmn,
                         double *source_corr,
                         int x0_start, int x0_end, double w) {

    struct recombine2d_config *cfg = &wcfg->recombine;
    int offset = sizeof(double complex) *x0_start * cfg->yB_size;
    int size = sizeof(double complex) *(x0_end - x0_start) * cfg->yB_size;

    if (work->path && !work->hdf5) {

        printf("Reading facet data from %s (%d-%d)...\n", work->path, x0_start, x0_end);

        // Make sure strides are compatible
        assert (cfg->F_stride0 == cfg->yB_size && cfg->F_stride1 == 1);

        // Load data from file
        int fd = open(work->path, O_RDONLY, 0666);
        if (fd > 0) {
            lseek(fd, offset, SEEK_SET);
            if (read(fd, F, size) != size) {
                fprintf(stderr, "failed to read enough data from %s for range %d-%d!\n", work->path, x0_start, x0_end);
                return false;
            }
            close(fd);
        } else {
            fprintf(stderr, "Failed to read facet data!\n");
        }

    } else if (work->path && work->hdf5) {

        printf("Reading facet data from %s:%s (%d-%d)...\n", work->hdf5, work->path, x0_start, x0_end);

        // Make sure strides are as expected, then read
        // TODO: Clearly HDF5 can do partial reads, optimise
        assert (cfg->F_stride0 == cfg->yB_size && cfg->F_stride1 == 1);
        double complex *data;
#pragma omp critical
        data = read_hdf5(cfg->F_size, work->hdf5, work->path);

        // Copy
        memcpy(F, data + offset / sizeof(double complex), size);
        free(data);

    } else if (source_count > 0) {

        // Place sources in gridder's usable region
        int i;
        for (i = 0; i < source_count; i++) {
            int il = source_xy[i*2+0], im = source_xy[i*2+1];

            // Skip sources outside the current facet (region)
            if (il - work->facet_off_l < -cfg->yB_size/2 ||
                il - work->facet_off_l >= cfg->yB_size/2 ||
                im - work->facet_off_m < -cfg->yB_size/2 ||
                im - work->facet_off_m >= cfg->yB_size/2) {

                continue;
            }

            // Calculate facet coordinates, keeping in mind that the
            // centre is at (0/0).
            int x0 = (im - work->facet_off_m + cfg->yB_size) % cfg->yB_size;
            int x1 = (il - work->facet_off_l + cfg->yB_size) % cfg->yB_size;
            if (x0 < x0_start || x0 >= x0_end) {
                continue;
            }

            // Calculate Fresnel pattern for w-stacking
            complex double fresnel = 1;
            if (w != 0) {
                double ph = w * source_lmn[i*3+2];
                fresnel = cos(2*M_PI*ph) + 1.j * sin(2*M_PI*ph);
            }

            // Add source, with gridding correction applied
            F[(x0-x0_start)*cfg->F_stride0 + x1*cfg->F_stride1]
                += fresnel / source_corr[i];
        }

    } else {

        // Fill facet with deterministic pseudo-random numbers
        int x0, x1;
        for (x0 = x0_start; x0 < x0_end; x0++) {
            unsigned int seed = x0;
            for (x1 = 0; x1 < cfg->yB_size; x1++) {
                F[(x0-x0_start)*cfg->F_stride0+x1*cfg->F_stride1] = (double)rand_r(&seed) / RAND_MAX;
            }
        }

    }
    return true;
}

// Gets subgrid offset for given column/rpw. Returns INT_MIN if no work was found.
static int get_subgrid_off_u(struct work_config *wcfg, int iu, int iw)
{

    // Somewhat inefficiently walk entire work list
    int iwork;
    for (iwork = 0; iwork < wcfg->subgrid_workers * wcfg->subgrid_max_work; iwork++) {
        if (wcfg->subgrid_work[iwork].nbl > 0 &&
            wcfg->subgrid_work[iwork].iu == iu &&
            wcfg->subgrid_work[iwork].iw == iw) break;
    }
    if (iwork >= wcfg->subgrid_workers * wcfg->subgrid_max_work)
        return INT_MIN;

    return wcfg->subgrid_work[iwork].subgrid_off_u;
}

static int get_subgrid_off_v(struct work_config *wcfg, int iu, int iv, int iw)
{

    // Somewhat inefficiently walk entire work list
    int iwork;
    for (iwork = 0; iwork < wcfg->subgrid_workers * wcfg->subgrid_max_work; iwork++) {
        if (wcfg->subgrid_work[iwork].nbl > 0 &&
            wcfg->subgrid_work[iwork].iu == iu &&
            wcfg->subgrid_work[iwork].iv == iv &&
            wcfg->subgrid_work[iwork].iw == iw) break;
    }
    if (iwork >= wcfg->subgrid_workers * wcfg->subgrid_max_work) return INT_MIN;

    return wcfg->subgrid_work[iwork].subgrid_off_v;
}


// Produce subgrid data from facet data at some w-level
static void producer_facets_work(struct work_config *wcfg,
                                 struct producer_stream *prod,
                                 struct producer_stream *producers,
                                 double complex *F, double complex *BF,
                                 int wlevel)
{

    int ifacet;

    // Do first stage preparation and Fourier Transform
    if (wcfg->produce_retain_bf)
        for (ifacet = 0; ifacet < prod->facet_work_count; ifacet++)
            recombine2d_pf1_ft1_omp(&prod->worker,
                                    F + ifacet * wcfg->recombine.F_size / sizeof(*F),
                                    BF + ifacet * wcfg->recombine.BF_size / sizeof(*BF));

    int iu;
    if (wcfg->produce_parallel_cols) {
        // Go through columns in parallel
        #pragma omp for schedule(dynamic)
        for (iu = wcfg->iu_min; iu <= wcfg->iu_max ; iu++) {

            // Determine column offset / check whether column actually has work
            int subgrid_off_u = get_subgrid_off_u(wcfg, iu, wlevel);
            if (subgrid_off_u == INT_MIN) continue;

            // Loop through facets sequentially (inefficient, as it
            // introduces a time delay on when we touch facets)
            for (ifacet = 0; ifacet < prod->facet_work_count; ifacet++) {

                // Extract subgrids along first axis, then prepare and Fourier
                // transform along second axis
                recombine2d_es1_pf0_ft0(&prod->worker, subgrid_off_u,
                                        BF + ifacet * wcfg->recombine.BF_size / sizeof(*BF),
                                        prod->worker.NMBF_BF);

                // Go through rows in sequence
                int iv;
                for (iv = wcfg->iv_min; iv <= wcfg->iv_max; iv++) {
                    int subgrid_off_v = get_subgrid_off_v(wcfg, iu, iv, wlevel);
                    if (subgrid_off_v == INT_MIN) continue;
                    producer_send_subgrid(wcfg, prod, ifacet, prod->worker.NMBF_BF,
                                          subgrid_off_u, subgrid_off_v, iu, iv,
                                          wlevel);
                }
            }
        }
    } else {
        // Go through columns in sequence
        for (iu = wcfg->iu_min; iu <= wcfg->iu_max; iu++) {

            // Determine column offset / check whether column actually has work
            int subgrid_off_u = get_subgrid_off_u(wcfg, iu, wlevel);
            if (subgrid_off_u == INT_MIN) continue;

            // Loop through facets (inefficient, see above)
            for (ifacet = 0; ifacet < prod->facet_work_count; ifacet++) {

                // Extract subgrids along first axis, then prepare and Fourier
                // transform along second axis
                double complex *NMBF = producers->worker.NMBF;
                double complex *NMBF_BF = producers->worker.NMBF_BF;
                if (wcfg->produce_retain_bf)
                    recombine2d_es1_omp(&prod->worker, subgrid_off_u,
                                        BF + ifacet * wcfg->recombine.BF_size / sizeof(*BF),
                                        NMBF);
                else
                    recombine2d_pf1_ft1_es1_omp(&prod->worker, subgrid_off_u,
                                                F + ifacet * wcfg->recombine.F_size / sizeof(*F),
                                                NMBF);
                recombine2d_pf0_ft0_omp(&prod->worker, NMBF, NMBF_BF);

                // Go through rows in parallel
                int iv;
                #pragma omp for schedule(dynamic)
                for (iv = wcfg->iv_min; iv <= wcfg->iv_max; iv++) {
                    int subgrid_off_v = get_subgrid_off_v(wcfg, iu, iv, wlevel);
                    if (subgrid_off_v == INT_MIN) continue;
                    producer_send_subgrid(wcfg, prod, ifacet, NMBF_BF,
                                          subgrid_off_u, subgrid_off_v, iu, iv,
                                          wlevel);
                }
            }
        }
    }
}

double producer_work(struct work_config *wcfg,
                     struct producer_stream *producers,
                     int facet_work_count,
                     double complex *F, double complex *BF)
{
    struct recombine2d_config *const cfg = &wcfg->recombine;
    struct producer_stream *const prod = producers + omp_get_thread_num();
    struct facet_work *const fwork = wcfg->facet_work +
        prod->facet_worker * wcfg->facet_max_work;

    // Determine first and last needed w-level from subgrid work list
    int min_wlevel = INT_MAX, max_wlevel = INT_MIN; int iwork;
    for (iwork = 0; iwork < wcfg->subgrid_workers * wcfg->subgrid_max_work;
         iwork++) {
        if (!wcfg->subgrid_work[iwork].nbl)
            continue;
        if (min_wlevel > wcfg->subgrid_work[iwork].iw)
            min_wlevel = wcfg->subgrid_work[iwork].iw;
        if (max_wlevel < wcfg->subgrid_work[iwork].iw)
            max_wlevel = wcfg->subgrid_work[iwork].iw;
    }

    int wlevel;
    double stream_time = 0;
    for (wlevel = min_wlevel; wlevel <= max_wlevel; wlevel++) {

        // Check whether wlevel should be skipped
        bool found = false;
        for (iwork = 0; iwork < wcfg->subgrid_workers * wcfg->subgrid_max_work;
             iwork++) {
            if (!wcfg->subgrid_work[iwork].nbl)
                continue;
            if (wcfg->subgrid_work[iwork].iw == wlevel) {
                found = true;
                break;
            }
        }
        if (!found)
            continue;

        // Start of facet data creation
        double generate_start;
        #pragma omp single copyprivate(generate_start)
        {
            printf("Filling %d facet%s (w level %d)...\n",
                   facet_work_count, facet_work_count != 1 ? "s" : "",
                   wlevel);
            generate_start = get_time_ns();
        }

        // Parallelise over facets and facet chunks
        int ifacet; int x0; const int x0_chunk = 256;
        #pragma omp for schedule(dynamic) collapse(2)
        for (ifacet = 0; ifacet < facet_work_count; ifacet++) {
            for (x0 = 0; x0 < cfg->yB_size; x0+=x0_chunk) {
                int x0_end = x0 + x0_chunk;
                if (x0_end > cfg->yB_size) x0_end = cfg->yB_size;
                double complex *pF =
                    F + ifacet * wcfg->recombine.F_size / sizeof(*F)
                    + x0*cfg->F_stride0;
                memset(pF, 0, sizeof(*pF) * cfg->F_stride0 *
                       (x0_end-x0 > x0_chunk ? x0_chunk : x0_end-x0));

                double w = wlevel * wcfg->wstep * wcfg->sg_step_w;
                producer_fill_facet(wcfg, fwork + ifacet, pF,
                                    wcfg->source_count, wcfg->source_xy,
                                    wcfg->source_lmn, wcfg->source_corr,
                                    x0, x0_end, w);
            }
        }

        // Done creating facet data
        double run_start;
    #pragma omp single copyprivate(run_start)
        {
            printf(" %.2f s\n", get_time_ns() - generate_start);

            run_start = get_time_ns();
            printf("Streaming...\n");
        }

        // Start generating subgrid data
        producer_facets_work(wcfg, prod, producers, F, BF, wlevel);

#pragma omp single copyprivate(stream_time)
        {
            stream_time += get_time_ns() - run_start;
        }

    }

    return stream_time;
}

int producer(struct work_config *wcfg, int facet_worker, int *streamer_ranks)
{

    struct recombine2d_config *cfg = &wcfg->recombine;
    struct facet_work *fwork = wcfg->facet_work + facet_worker * wcfg->facet_max_work;

    const int BF_batch = wcfg->produce_batch_rows;
    const int send_queue_length = wcfg->produce_queue_length;

    // Get number of facets we need to cover
    int facet_work_count = 0; int ifacet;
    for (ifacet = 0; ifacet < wcfg->facet_max_work; ifacet++)
        if (fwork[ifacet].set)
            facet_work_count++;

    // Determine required buffer sizes. If we don't retain the full
    // padded facet, we still need enough space to be able to work on
    // one batch of rows.
    uint64_t F_size = facet_work_count * cfg->F_size;
    uint64_t BF_size = wcfg->produce_retain_bf ?
        facet_work_count * cfg->BF_size :
        sizeof(double complex) * cfg->yP_size * BF_batch;

    printf("Using %.1f GB global, %.1f GB per thread\n",
           (double)(F_size + BF_size) / 1000000000,
           facet_work_count * (double)recombine2d_worker_memory(cfg) / 1000000000);

    // Create global memory buffers for facet at current w-level
    double complex *F = (double complex *)calloc(1, F_size);
    double complex *BF = (double complex *)malloc(BF_size);
    if (!F || (!BF && wcfg->produce_retain_bf)) {
        free(F); free(BF);
        printf("Failed to allocate global buffers!\n");
        return 1;
    }


    // Global structures
    double stream_time;
    int producer_count;
    struct producer_stream *producers;

    #pragma omp parallel
    {

        // Perform planning (need to know thread count for that)
        producer_count = omp_get_num_threads();
        #pragma omp single
        {
            // Do global planning
            printf("Planning for %d threads...\n", producer_count);
            double planning_start = get_time_ns();
            fftw_plan BF_plan = recombine2d_bf_plan(cfg, BF_batch,
                                                    BF, FFTW_MEASURE);

            // Create producers (which involves planning, and
            // therefore is not parallelised)
            producers = (struct producer_stream *) malloc(
                sizeof(struct producer_stream) * producer_count);
            int i;
            for (i = 0; i < producer_count; i++) {
                init_producer_stream(cfg, producers + i, facet_worker, facet_work_count,
                                     wcfg->facet_workers, streamer_ranks,
                                     BF_batch, BF_plan, send_queue_length);
            }

            printf(" %.2f s\n", get_time_ns() - planning_start);

        }

        // Start creating facets and streaming subgrid data out
        stream_time = producer_work(wcfg, producers, facet_work_count, F, BF);

#ifndef NO_MPI
        // Wait for remaining packets to be sent
        struct producer_stream *prod = producers + omp_get_thread_num();
        double start = get_time_ns();
        MPI_Status statuses[send_queue_length];
        MPI_Waitall(send_queue_length, prod->requests, statuses);
        prod->mpi_wait_time += get_time_ns() - start;
#endif

        free_producer_stream(prod);
    }
    free(BF);
    free(F);

    fftw_free(producers[0].worker.BF_plan);

    // Show statistics
    int p;
    for (p = 1; p < producer_count; p++) {
        producer_add_stats(producers, producers + p);
    }
    producer_dump_stats(wcfg, facet_worker,
                        producers, producer_count, stream_time);

    return 0;
}
