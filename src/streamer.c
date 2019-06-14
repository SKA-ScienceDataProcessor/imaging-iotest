
#include "grid.h"
#include "config.h"

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
#include <semaphore.h>

#ifndef NO_MPI
#include <mpi.h>
#else
#define MPI_Request int
#define MPI_REQUEST_NULL 0
#endif

struct streamer_chunk
{
    int a1, a2, tchunk, fchunk;
    sem_t in_lock, out_lock; // Ready to fill / write to disk
    double complex *vis;
};

struct streamer_writer
{

    // Index of worker, and writer (unique across distributed program)
    int subgrid_worker;
    int index;
    pid_t pid; // if forking writers
    pthread_t thread; // if not forking writers

    // Visibility file
    hid_t file, group;

    // Visibility Chunk queue
    int queue_length;
    int in_ptr, out_ptr;
    int to_write;
    struct streamer_chunk *queue;

    // Work config
    struct work_config *work_cfg;

    // Statistics
    double wait_out_time;
    double read_time;
    double write_time;
    uint64_t written_vis_data, rewritten_vis_data;

};

struct streamer
{
    struct work_config *work_cfg;
    int subgrid_worker;
    int *producer_ranks;

    struct sep_kernel_data kern;
    bool have_kern;

    // Incoming data queue (to be assembled)
    int queue_length;
    double complex *nmbf_queue;
    MPI_Request *request_queue;
    int *request_work; // per request: subgrid work to perform
    bool *skip_receive; // per subgrid work: skip, because subgrid is being/was received already

    // Subgrid queue (to be degridded)
    double complex *subgrid_queue;
    int subgrid_tasks;
    int *subgrid_locks;
    fftw_plan subgrid_plan;

    // Visibility chunk queue (to be written)
    int writer_count;
    int vis_queue_length;
    size_t vis_queue_size, vis_chunks_size, writer_size;
    int vis_queue_per_writer;
    double complex *vis_queue;
    struct streamer_chunk *vis_chunks;
    struct streamer_writer *writer;

    // Statistics
    int num_workers;
    double wait_time;
    double wait_in_time;
    double recombine_time, task_start_time, degrid_time;
    uint64_t received_data, received_subgrids, baselines_covered;
    uint64_t square_error_samples;
    double square_error_sum, worst_error;
    uint64_t degrid_flops;
    uint64_t produced_chunks;
    uint64_t task_yields;

    // Signal for being finished
    bool finished;
};

static double complex *nmbf_slot(struct streamer *streamer, int slot, int facet)
{
    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    const int xM_yN_size = streamer->work_cfg->recombine.xM_yN_size;
    assert(facet >= 0 && facet < (facets == 0 ? 1 : facets));
    return streamer->nmbf_queue + xM_yN_size * xM_yN_size * ((slot * facets) + facet);
}

static MPI_Request *request_slot(struct streamer *streamer, int slot, int facet)
{
    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    assert(facet >= 0 && facet < (facets == 0 ? 1 : facets));
    return streamer->request_queue + (slot * facets) + facet;
}

static double complex *subgrid_slot(struct streamer *streamer, int slot)
{
    const int xM_size = streamer->work_cfg->recombine.xM_size;
    return streamer->subgrid_queue + xM_size * xM_size * slot;
}

struct streamer_chunk *writer_push_slot(struct streamer_writer *writer,
                                        int a1, int a2, int tchunk, int fchunk)
{
    if (!writer) return NULL;

    // Determine our slot (competing with other tasks, so need to have
    // critical section here)
    int vis_slot;
    #pragma omp critical
    {
        vis_slot = writer->in_ptr;
        writer->in_ptr = (writer->in_ptr + 1) % writer->queue_length;
    }

    // Obtain lock for writing data (writer thread might not have
    // written this data to disk yet)
    struct streamer_chunk *chunk = writer->queue + vis_slot;
#pragma omp atomic
    writer->to_write += 1;
    sem_wait(&chunk->in_lock);

    // Set slot data
    chunk->a1 = a1;
    chunk->a2 = a2;
    chunk->tchunk = tchunk;
    chunk->fchunk = fchunk;
    return chunk;
}

void streamer_ireceive(struct streamer *streamer,
                       int subgrid_work, int slot)
{

    const int xM_yN_size = streamer->work_cfg->recombine.xM_yN_size;
    struct subgrid_work *work = streamer->work_cfg->subgrid_work +
        streamer->subgrid_worker * streamer->work_cfg->subgrid_max_work;
    const int facet_count = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;

    // Not populated or marked to skip? Clear slot
    if (!work[subgrid_work].nbl || streamer->skip_receive[subgrid_work]) {
        int facet;
        for (facet = 0; facet < facet_count; facet++) {
            *request_slot(streamer, slot, facet) = MPI_REQUEST_NULL;
        }
        streamer->request_work[slot] = -1;
        return;
    }

    // Set work
    streamer->request_work[slot] = subgrid_work;

    // Mark later subgrid repeats for skipping
    int iw;
    for (iw = subgrid_work+1; iw < streamer->work_cfg->subgrid_max_work; iw++)
        if (work[iw].iu == work[subgrid_work].iu && work[iw].iv == work[subgrid_work].iv)
            streamer->skip_receive[iw] = true;

    // Walk through all facets we expect contributions from, save requests
    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    int facet;
    for (facet = 0; facet < facets; facet++) {
        struct facet_work *fwork = streamer->work_cfg->facet_work + facet;
        if (!fwork->set) {
            *request_slot(streamer, slot, facet) = MPI_REQUEST_NULL;
            continue;
        }

#ifndef NO_MPI
        // Set up a receive slot with appropriate tag
        const int facet_worker = facet / streamer->work_cfg->facet_max_work;
        const int facet_work = facet % streamer->work_cfg->facet_max_work;
        const int tag = make_subgrid_tag(streamer->work_cfg,
                                         streamer->subgrid_worker, subgrid_work,
                                         facet_worker, facet_work);
        MPI_Irecv(nmbf_slot(streamer, slot, facet),
                  xM_yN_size * xM_yN_size, MPI_DOUBLE_COMPLEX,
                  streamer->producer_ranks[facet_worker], tag, MPI_COMM_WORLD,
                  request_slot(streamer, slot, facet));
#endif
    }

}

void streamer_work(struct streamer *streamer,
                   int subgrid_work,
                   double complex *nmbf);

int streamer_receive_a_subgrid(struct streamer *streamer,
                               struct subgrid_work *work,
                               int *waitsome_indices)
{
    const int facet_work_count = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;

    double start = get_time_ns();

    int slot; int waiting = 0;
    for(;;) {

        // Wait for MPI data to arrive. We only use "Waitsome" if this
        // is not the first iteration *and* we know that there are
        // actually waiting requests. Waitsome might block
        // indefinetely - so we should only do that if we are sure
        // that we need to receive *something* before we get more work
        // to do.
        int index_count = 0;
        if (facet_work_count > 0) {
            if (waiting == 0) {
                MPI_Testsome(facet_work_count * streamer->queue_length, streamer->request_queue,
                             &index_count, waitsome_indices, MPI_STATUSES_IGNORE);
            } else {
                MPI_Waitsome(facet_work_count * streamer->queue_length, streamer->request_queue,
                             &index_count, waitsome_indices, MPI_STATUSES_IGNORE);
            }
        }

        // Note how much data was received
        int i;
        if (index_count > 0) {
#pragma omp atomic
            streamer->received_data += index_count * streamer->work_cfg->recombine.NMBF_NMBF_size;
        }

        // Check that the indicated requests were actually cleared
        // (this behaviour isn't exactly prominently documented?).
        for (i = 0; i < index_count; i++) {
            assert(streamer->request_queue[waitsome_indices[i]] == MPI_REQUEST_NULL);
            streamer->request_queue[waitsome_indices[i]] = MPI_REQUEST_NULL;
        }

        // Find finished slot
        waiting = 0;
        for (slot = 0; slot < streamer->queue_length; slot++) {
            if (streamer->request_work[slot] >= 0) {
                // Check whether all requests are finished
                int i;
                for (i = 0; i < facet_work_count; i++)
                    if (*request_slot(streamer, slot, i) != MPI_REQUEST_NULL)
                        break;
                if (i >= facet_work_count)
                    break;
                else {
                    waiting++;
                }
            }
        }

        // Found and task queue not full?
        if (streamer->subgrid_tasks < streamer->work_cfg->vis_task_queue_length) {
            if (slot < streamer->queue_length)
                break;
            assert(waiting > 0);
        } else {
            // We are waiting for tasks to finish - idle for a bit
            usleep(1000);
        }

    }

    // Alright, found a slot with all data to form a subgrid.
    double complex *data_slot = nmbf_slot(streamer, slot, 0);
    streamer->received_subgrids++;
    streamer->wait_time += get_time_ns() - start;

    // Do work on received data. If a subgrid appears twice in our
    // work list, spawn all of the matching work
    const int iwork = streamer->request_work[slot];
    int iw;
    for (iw = iwork; iw < streamer->work_cfg->subgrid_max_work; iw++)
        if (work[iw].iu == work[iwork].iu && work[iw].iv == work[iwork].iv)
            streamer_work(streamer, iw, data_slot);

    // Return the (now free) slot
    streamer->request_work[slot] = -1;
    return slot;
}

void *streamer_reader_thread(void *param)
{

    struct streamer *streamer = (struct streamer *)param;

    struct work_config *wcfg = streamer->work_cfg;
    struct subgrid_work *work = wcfg->subgrid_work + streamer->subgrid_worker * wcfg->subgrid_max_work;
    const int facets = wcfg->facet_workers * wcfg->facet_max_work;

    int *waitsome_indices = (int *)malloc(sizeof(int) * facets * streamer->queue_length);

    // Walk through subgrid work packets
    int iwork = 0, iwork_r = streamer->queue_length;
    for (iwork = 0; iwork < streamer->work_cfg->subgrid_max_work; iwork++) {

        // Skip?
        if (!work[iwork].nbl || streamer->skip_receive[iwork])
            continue;

        // Receive a subgrid. Does not have to be in order.
        int slot = streamer_receive_a_subgrid(streamer, work, waitsome_indices);

        // Set up slot for new data (if appropriate)
        while (iwork_r < wcfg->subgrid_max_work && !work[iwork_r].nbl)
            iwork_r++;
        if (iwork_r < wcfg->subgrid_max_work) {
            streamer_ireceive(streamer, iwork_r, slot);
        }
        iwork_r++;

    }

    free(waitsome_indices);

    return NULL;
}

void *streamer_writer_thread(void *param)
{
    struct streamer_writer *writer = (struct streamer_writer *) param;
    struct work_config *wcfg = writer->work_cfg;

    struct vis_spec *const spec = &writer->work_cfg->spec;
    const int vis_data_size = sizeof(double complex) * spec->time_chunk * spec->freq_chunk;
    double complex *vis_data_h5 = (double complex *) alloca(vis_data_size);

    // Create HDF5 output file if we are meant to output any amount of
    // visibilities
    if (!wcfg->vis_path)
        return NULL;

    // Get filename to use
    char filename[512];
    sprintf(filename, wcfg->vis_path, writer->index);

    // For some reason we need to protect creating the file with a
    // critical section, otherwise libhdf5 messes up. Note that
    // creating groups (below) is apparently fine to do in parallel.
#pragma omp critical
    {
        // Open file and "vis" group
        if (wcfg->vis_check_existing) {
            printf("\nOpening %s... ", filename);
            writer->file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
            writer->group = H5Gopen(writer->file, "vis", H5P_DEFAULT);
        } else {
            printf("\nCreating %s... ", filename);
            writer->file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            if (writer->file >= 0)
                writer->group = H5Gcreate(writer->file, "vis", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            // Create baseline groups
            if (writer->file >= 0 && writer->group >= 0)
                create_bl_groups(writer->group, wcfg, writer->subgrid_worker);
        }
    }

    if (writer->file < 0 || writer->group < 0) {
        fprintf(stderr, "Could not open visibility file %s!\n", filename);
        return NULL;
    }

    int time_chunk_count = spec->time_count / spec->time_chunk;
    int freq_chunk_count = spec->freq_count / spec->freq_chunk;
    int chunk_count = spec->cfg->ant_count * spec->cfg->ant_count
        * time_chunk_count * freq_chunk_count;
    bool *chunks_written = calloc(sizeof(bool), chunk_count);

    for(;;) {

        // Obtain "out" lock for writing out visibilities
        double start = get_time_ns();
        struct streamer_chunk *chunk = writer->queue + writer->out_ptr;
        sem_wait(&chunk->out_lock);
        writer->wait_out_time += get_time_ns() - start;

        start = get_time_ns();

        // Obtain baseline data
        struct bl_data bl_data;
        if (chunk->tchunk == -1 && chunk->fchunk == -1)
            break; // Signal to end thread
        if (chunk->tchunk == -2 && chunk->fchunk == -2) {
            #pragma omp atomic
                writer->to_write -= 1;
            sem_post(&chunk->in_lock);
            writer->out_ptr = (writer->out_ptr + 1) % writer->queue_length;
            continue; // Signal to ignore chunk
        }
        vis_spec_to_bl_data(&bl_data, spec, chunk->a1, chunk->a2);
        double complex *vis_data = chunk->vis;

        // Read visibility chunk. If it was not yet set, this will
        // just fill the buffer with zeroes.
        int chunk_index = ((bl_data.antenna2 * spec->cfg->ant_count + bl_data.antenna1)
                           * time_chunk_count + chunk->tchunk) * freq_chunk_count + chunk->fchunk;
        if (wcfg->vis_check_existing || chunks_written[chunk_index]) {
            read_vis_chunk(writer->group, &bl_data,
                           spec->time_chunk, spec->freq_chunk,
                           chunk->tchunk, chunk->fchunk,
                           vis_data_h5);
        } else {
            memset(vis_data_h5, 0, vis_data_size);
        }
        writer->read_time += get_time_ns() - start;
        start = get_time_ns();

        if (wcfg->vis_check_existing) {

            // Compare data
            int i;
            for (i = 0; i < spec->time_chunk * spec->freq_chunk; i++) {
                if (vis_data[i] != 0) {
                    if (cabs(vis_data_h5[i] - vis_data[i]) > 1e-12) {
                        printf("%g%+gj != %g%+gj (diff %g)!\n",
                               creal(vis_data_h5[i]), cimag(vis_data_h5[i]),
                               creal(vis_data[i]), cimag(vis_data[i]),
                               cabs(vis_data_h5[i] - vis_data[i]));
                    }
                }
            }
            writer->rewritten_vis_data += vis_data_size;

        } else {

            // Copy over data
            start = get_time_ns();
            int i;
            for (i = 0; i < spec->time_chunk * spec->freq_chunk; i++) {
                if (vis_data[i] != 0) {
                    // Make sure we never over-write data!
                    assert(vis_data_h5[i] == 0);
                    vis_data_h5[i] = vis_data[i];
                }
            }

            // Write chunk back
            write_vis_chunk(writer->group, &bl_data,
                            spec->time_chunk, spec->freq_chunk,
                            chunk->tchunk, chunk->fchunk,
                            vis_data_h5);

            writer->written_vis_data += vis_data_size;
            if (chunks_written[chunk_index])
                writer->rewritten_vis_data += vis_data_size;
            chunks_written[chunk_index] = true;

        }

        free(bl_data.time); free(bl_data.uvw_m); free(bl_data.freq);

        // Release "in" lock to mark the slot free for writing
        #pragma omp atomic
            writer->to_write -= 1;
        sem_post(&chunk->in_lock);
        writer->out_ptr = (writer->out_ptr + 1) % writer->queue_length;
        writer->write_time += get_time_ns() - start;

    }

    H5Gclose(writer->group); H5Fclose(writer->file);

    return NULL;
}

static double min(double a, double b) { return a > b ? b : a; }
static double max(double a, double b) { return a < b ? b : a; }

uint64_t streamer_degrid_worker(struct streamer *streamer,
                                struct bl_data *bl_data,
                                int SG_stride, double complex *subgrid,
                                double mid_u, double mid_v, int iu, int iv, bool conjugate,
                                int it0, int it1, int if0, int if1,
                                double min_u, double max_u, double min_v, double max_v,
                                double complex *vis_data)
{
    struct vis_spec *const spec = &streamer->work_cfg->spec;
    const double theta = streamer->work_cfg->theta;
    const int subgrid_size = streamer->work_cfg->recombine.xM_size;

    // Calculate l/m positions of sources in case we are meant to
    // check them
    int i;
    const int image_size = streamer->work_cfg->recombine.image_size;
    const int source_count = streamer->work_cfg->produce_source_count;
    int source_checks = streamer->work_cfg->produce_source_checks;
    const double facet_x0 = streamer->work_cfg->spec.fov / streamer->work_cfg->theta / 2;
    const int image_x0_size = (int)floor(2 * facet_x0 * image_size);
    double *source_pos_l = NULL, *source_pos_m = NULL;
    int check_counter = 0;
    if (source_checks > 0 && source_count > 0) {
        source_pos_l = (double *)malloc(sizeof(double) * source_count);
        source_pos_m = (double *)malloc(sizeof(double) * source_count);
        unsigned int seed = 0;
        for (i = 0; i < source_count; i++) {
            int il = (int)(rand_r(&seed) % image_x0_size) - image_x0_size / 2;
            int im = (int)(rand_r(&seed) % image_x0_size) - image_x0_size / 2;
            // Create source positions scaled for quick usage below
            source_pos_l[i] = 2 * M_PI * il * theta / image_size;
            source_pos_m[i] = 2 * M_PI * im * theta / image_size;
        }
        // Initialise counter to random value so we check random
        // visibilities
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
        double u = uvw_lambda(bl_data, time, 0, 0);
        double v = uvw_lambda(bl_data, time, 0, 1);
        double du = uvw_lambda(bl_data, time, 1, 0) - u;
        double dv = uvw_lambda(bl_data, time, 1, 1) - v;
        if (conjugate) {
          u *= -1; du *= -1; v *= -1; dv *= -1;
        }

        // Degrid a line of visibilities
        double complex *pvis = vis_data + (time-it0)*spec->freq_chunk;
        degrid_conv_uv_line(subgrid, subgrid_size, SG_stride, theta,
                            u-mid_u, v-mid_v, du, dv, if1 - if0,
                            min_u-mid_u, max_u-mid_u, min_v-mid_v, max_v-mid_v, conjugate,
                            &streamer->kern, pvis, &flops);

        // Check against DFT (one per row, maximum)
        if (source_checks > 0) {
            if (check_counter >= if1 - if0) {
                check_counter -= if1 - if0;

            } else {

                double complex vis_out = pvis[check_counter];
                double check_u = u + du * check_counter;
                double check_v = v + dv * check_counter;

                // Check that we actually generated a visibility here,
                // negate if necessary
                complex double vis = 0;
                if (check_u >= min_u && check_u < max_u &&
                    check_v >= min_v && check_v < max_v) {
                    if (conjugate) { check_u *= -1; check_v *= -1; }

                    // Generate visibility
                    for (i = 0; i < source_count; i++) {
                        double ph = check_u * source_pos_l[i] + check_v * source_pos_m[i];
                        vis += cos(ph) + 1.j * sin(ph);
                    }
                    vis /= (double)image_size * image_size;

                }

                // Check error
                square_error_samples += 1;
                double err = cabs(vis_out - vis);
                if (err > 1e-8 && false) {
                    fprintf(stderr,
                           "WARNING: uv %g/%g (sg %d/%d): %g%+gj != %g%+gj\n",
                           u, v, iu, iv, creal(vis_out), cimag(vis_out), creal(vis), cimag(vis));
                }
                worst_err = fmax(err, worst_err);
                square_error_sum += err * err;

                check_counter = source_checks;
            }
        }
    }

    free(source_pos_l); free(source_pos_m);

    // Add to statistics
    #pragma omp atomic
        streamer->square_error_samples += square_error_samples;
    #pragma omp atomic
        streamer->square_error_sum += square_error_sum;
    if (worst_err > streamer->worst_error) {
        // Likely happens rarely enough that a critical section won't be a problem
        #pragma omp critical
           streamer->worst_error = max(streamer->worst_error, worst_err);
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
                           struct bl_data *bl_data,
                           int tchunk, int fchunk,
                           int slot,
                           int SG_stride, double complex *subgrid)
{
    struct vis_spec *const spec = &streamer->work_cfg->spec;
    struct recombine2d_config *const cfg = &streamer->work_cfg->recombine;
    const double theta = streamer->work_cfg->theta;

    double start = get_time_ns();

    // Calculate subgrid boundaries. TODO: All of this duplicates
    // logic that also appears in config.c (bin_baseline). This is
    // brittle, should get refactored at some point!
    double sg_mid_u = work->subgrid_off_u / theta;
    double sg_mid_v = work->subgrid_off_v / theta;
    double sg_min_u = (work->subgrid_off_u - cfg->xA_size / 2) / theta;
    double sg_min_v = (work->subgrid_off_v - cfg->xA_size / 2) / theta;
    double sg_max_u = (work->subgrid_off_u + cfg->xA_size / 2) / theta;
    double sg_max_v = (work->subgrid_off_v + cfg->xA_size / 2) / theta;
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

    // Check whether we actually have overlap with the subgrid (TODO:
    // again slightly naive, there's a chance we lose some
    // visibilities here)
    double f0 = uvw_m_to_l(1, bl_data->freq[if0]);
    double f1 = uvw_m_to_l(1, bl_data->freq[if1-1]);
    double *uvw0 = bl_data->uvw_m + 3*it0;
    double *uvw1 = bl_data->uvw_m + 3*(it1-1);
    double min_u = min(min(uvw0[0]*f0, uvw0[0]*f1), min(uvw1[0]*f0, uvw1[0]*f1));
    double min_v = min(min(uvw0[1]*f0, uvw0[1]*f1), min(uvw1[1]*f0, uvw1[1]*f1));
    double max_u = max(max(uvw0[0]*f0, uvw0[0]*f1), max(uvw1[0]*f0, uvw1[0]*f1));
    double max_v = max(max(uvw0[1]*f0, uvw0[1]*f1), max(uvw1[1]*f0, uvw1[1]*f1));

    // Check whether time chunk fall into positive u. We use this
    // for deciding whether coordinates are going to get flipped
    // for the entire chunk. This is assuming that a chunk is
    // never big enough that we would overlap an extra subgrid
    // into the negative direction.
    int tstep_mid = (it0 + it1) / 2;
    double uvw_mid[3];
    ha_to_uvw_sc(spec->cfg, bl->a1, bl->a2,
                 spec->ha_sin[tstep_mid], spec->ha_cos[tstep_mid],
                 spec->dec_sin, spec->dec_cos,
                 uvw_mid);
    bool positive_u = uvw_mid[0] >= 0;
    if (!positive_u) {
        double swap;
        swap = min_u; min_u = -max_u; max_u = -swap;
        swap = min_v; min_v = -max_v; max_v = -swap;
    }

    // Check for overlap between baseline chunk and subgrid
    if (!(min_u < sg_max_u && max_u > sg_min_u &&
          min_v < sg_max_v && max_v > sg_min_v))
        return false;

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
        = writer_push_slot(writer, bl->a1, bl->a2, tchunk, fchunk);
    #pragma omp atomic
        streamer->wait_in_time += get_time_ns() - start;
    start = get_time_ns();

    // Do degridding
    const size_t chunk_vis_size = sizeof(double complex) * spec->freq_chunk * spec->time_chunk;
    uint64_t flops = streamer_degrid_worker(streamer, bl_data, SG_stride, subgrid,
                                            sg_mid_u, sg_mid_v, work->iu, work->iv, !positive_u,
                                            it0, it1, if0, if1,
                                            sg_min_u, sg_max_u, sg_min_v, sg_max_v,
                                            chunk ? chunk->vis : alloca(chunk_vis_size));
    #pragma omp atomic
      streamer->degrid_time += get_time_ns() - start;

    if (chunk) {

        // No flops executed? Signal to writer that we can skip writing
        // this chunk (small optimisation)
        if (flops == 0) {
            chunk->tchunk = -2;
            chunk->fchunk = -2;
        }

        // Signal slot for output
        sem_post(&chunk->out_lock);
    }

    return true;
}

void streamer_task(struct streamer *streamer,
                   struct subgrid_work *work,
                   struct subgrid_work_bl *bl,
                   int slot,
                   int subgrid_work,
                   double complex *subgrid)
{


    const int xM_size = streamer->work_cfg->recombine.xM_size;
    const int SG_stride = xM_size + 16; // Assume 16x16 is biggest possible convolution
    const int SG2_size = sizeof(double complex) *
        SG_stride * xM_size;
    double complex *subgrid2 = calloc(1, SG2_size);
    int i;

    // Establish proper stride for the subgrid so we don't get cache
    // thrashing problems when gridding moves (TODO: construct like this right away?)
    if (subgrid)
        for (i = 0; i < xM_size; i++) {
            memcpy(subgrid2 + SG_stride * i,
                   subgrid + xM_size * i,
                   sizeof(double complex) * xM_size);
        }

    struct vis_spec *const spec = &streamer->work_cfg->spec;
    struct subgrid_work_bl *bl2;
    int i_bl2;
    for (bl2 = bl, i_bl2 = 0;
         bl2 && i_bl2 < streamer->work_cfg->vis_bls_per_task;
         bl2 = bl2->next, i_bl2++) {

        // Get baseline data
        struct bl_data bl_data;
        vis_spec_to_bl_data(&bl_data, spec, bl2->a1, bl2->a2);

        // Go through time/frequency chunks
        int ntchunk = (bl_data.time_count + spec->time_chunk - 1) / spec->time_chunk;
        int nfchunk = (bl_data.freq_count + spec->freq_chunk - 1) / spec->freq_chunk;
        int tchunk, fchunk;
        int nchunks = 0;
        for (tchunk = 0; tchunk < ntchunk; tchunk++)
            for (fchunk = 0; fchunk < nfchunk; fchunk++)
                if (streamer_degrid_chunk(streamer, work,
                                          bl2, &bl_data, tchunk, fchunk,
                                          slot, SG_stride, subgrid2))
                    nchunks++;

        // Check that plan predicted the right number of chunks. This
        // is pretty important - if this fails this means that the
        // coordinate calculations are out of synch, which might mean
        // that we have failed to account for some visibilities in the
        // plan!
        if (bl2->chunks != nchunks)
            printf("WARNING: subgrid (%d/%d) baseline (%d-%d) %d chunks planned, %d actual!\n",
                   work->iu, work->iv, bl2->a1, bl2->a2, bl2->chunks, nchunks);

        free(bl_data.time); free(bl_data.uvw_m); free(bl_data.freq);
    }

    // Done with this chunk
#pragma omp atomic
    streamer->subgrid_tasks--;
#pragma omp atomic
    streamer->subgrid_locks[slot]--;

    free(subgrid2);
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

    // Perform Fourier transform
    fftw_execute_dft(streamer->subgrid_plan, subgrid, subgrid);

    // Check accumulated result
    if (work->check_path && streamer->work_cfg->facet_workers > 0) {
        double complex *approx_ref = read_hdf5(cfg->SG_size, work->check_hdf5, work->check_path);
        double err_sum = 0; int y;
        for (y = 0; y < cfg->xM_size * cfg->xM_size; y++) {
            double err = cabs(subgrid[y] - approx_ref[y]); err_sum += err * err;
        }
        free(approx_ref);
        double rmse = sqrt(err_sum / cfg->xM_size / cfg->xM_size);
        printf("%sSubgrid %d/%d RMSE %g\n", rmse > work->check_threshold ? "ERROR: " : "",
               work->iu, work->iv, rmse);

    }

    fft_shift(subgrid, cfg->xM_size);

    // Check some degridded example visibilities
    if (work->check_degrid_path && streamer->have_kern && streamer->work_cfg->facet_workers > 0) {
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
                       &bl, 0, nvis, 0, 1, &streamer->kern);
        double err_sum = 0; int y;
        for (y = 0; y < nvis; y++) {
            double err = cabs(vis[y] - bl.vis[y]); err_sum += err*err;
        }
        double rmse = sqrt(err_sum / nvis);
        printf("%sSubgrid %d/%d degrid RMSE %g\n",
               rmse > work->check_degrid_threshold ? "ERROR: " : "",
               work->iu, work->iv, rmse);

    }

    streamer->recombine_time += get_time_ns() - recombine_start;

    struct vis_spec *const spec = &streamer->work_cfg->spec;
    if (spec->time_count > 0 && streamer->have_kern) {

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

        printf("Subgrid %d/%d (%d baselines)\n", work->iu, work->iv, i_bl);
        fflush(stdout);
        streamer->baselines_covered += i_bl;

    }
}

static void _append_stat(char *stats, const char *stat_name, int worker, double val, double mult)
{
    sprintf(stats + strlen(stats), "user.recombine.%s:%g|g|#streamer:%d\n",
            stat_name, (double)(val * mult), worker);
}

static void _append_writer_stat(char *stats, const char *stat_name, int worker, int writer,
                                double val, double mult)
{
    sprintf(stats + strlen(stats), "user.recombine.%s:%g|g|#streamer:%d,writer:%d\n",
            stat_name, (double)(val * mult), worker, writer);
}

void *streamer_publish_stats(void *par)
{

    // Get streamer state. Make a copy to maintain differences
    struct streamer *streamer = (struct streamer *)par;
    struct streamer last, now;
    memcpy(&last, streamer, sizeof(struct streamer));

    struct streamer_writer *writers_last, *writers_now;
    const size_t writers_size = sizeof(struct streamer_writer) * streamer->writer_count;
    writers_last = (struct streamer_writer *)malloc(writers_size);
    writers_now = (struct streamer_writer *)malloc(writers_size);
    memcpy(writers_last, streamer->writer, writers_size);
    last.writer = writers_last;

    double sample_rate = streamer->work_cfg->statsd_rate;
    double next_stats = get_time_ns() + sample_rate;

    while(!streamer->finished) {

        // Make a snapshot of streamer state
        memcpy(&now, streamer, sizeof(struct streamer));
        memcpy(writers_now, streamer->writer, writers_size);
        now.writer = writers_now;

        // Add counters
        char stats[4096];
        stats[0] = 0;
#define PARS(stat) stats, #stat, streamer->subgrid_worker, now.stat - last.stat
        _append_stat(PARS(wait_time), 100. / sample_rate);
        _append_stat(PARS(wait_in_time), 100. / streamer->num_workers / sample_rate);
        _append_stat(PARS(task_start_time), 100 / sample_rate);
        _append_stat(PARS(recombine_time), 100 / sample_rate);
        _append_stat(PARS(degrid_time), 100. / streamer->num_workers / sample_rate);
        _append_stat(PARS(received_data), 1 / sample_rate);
        _append_stat(PARS(received_subgrids), 1 / sample_rate);
        _append_stat(PARS(baselines_covered), 1 / sample_rate);
        _append_stat(PARS(degrid_flops), 1 / sample_rate);
        _append_stat(PARS(produced_chunks), 1 / sample_rate);
        _append_stat(PARS(task_yields), 1 / sample_rate);
        _append_stat(PARS(square_error_samples), 1 / sample_rate);
#undef PARS
        config_send_statsd(streamer->work_cfg, stats);
        stats[0] = 0;

        int i;
        for (i = 0; i < streamer->writer_count; i++) {
#define PARS(stat) stats, #stat, streamer->subgrid_worker, now.writer[i].index, now.writer[i].stat - last.writer[i].stat
            _append_writer_stat(PARS(wait_out_time), 100 / sample_rate);
            _append_writer_stat(PARS(read_time), 100 / sample_rate);
            _append_writer_stat(PARS(write_time), 100 / sample_rate);
            _append_writer_stat(PARS(written_vis_data), 1 / sample_rate);
            _append_writer_stat(PARS(rewritten_vis_data), 1 / sample_rate);
#undef PARS
            _append_writer_stat(stats, "chunks_to_write", streamer->subgrid_worker,
                                now.writer[i].index, now.writer[i].to_write, 1);
        }

        // Receiver idle time
        double receiver_idle_time = 0;
        receiver_idle_time = sample_rate;
        receiver_idle_time -= now.wait_time - last.wait_time;
        receiver_idle_time -= now.task_start_time - last.task_start_time;
        receiver_idle_time -= now.recombine_time - last.recombine_time;
        _append_stat(stats, "receiver_idle_time", streamer->subgrid_worker,
                     receiver_idle_time, 100 / sample_rate);

        // Worker idle time
        double worker_idle_time = 0;
        worker_idle_time = streamer->num_workers * sample_rate;
        worker_idle_time -= now.wait_in_time - last.wait_in_time;
        worker_idle_time -= now.degrid_time - last.degrid_time;
        _append_stat(stats, "worker_idle_time", streamer->subgrid_worker,
                     worker_idle_time, 100 / sample_rate / streamer->num_workers);

        // Add gauges
        double derror_sum = now.square_error_sum - last.square_error_sum;
        uint64_t dsamples = now.square_error_samples - last.square_error_samples;
        if (dsamples > 0) {
            const int source_count = streamer->work_cfg->produce_source_count;
            const int image_size = streamer->work_cfg->recombine.image_size;
            double energy = (double)source_count / image_size / image_size;
            _append_stat(stats, "visibility_samples", streamer->subgrid_worker,
                         dsamples, 1);
            _append_stat(stats, "visibility_rmse", streamer->subgrid_worker,
                         10 * log(sqrt(derror_sum / dsamples) / energy) / log(10), 1);
        }
        _append_stat(stats, "degrid_tasks", streamer->subgrid_worker,
                     now.subgrid_tasks, 1);

        const int request_queue_length = streamer->work_cfg->facet_workers *
            streamer->work_cfg->facet_max_work * streamer->queue_length;
        int nrequests = 0;
        for (i = 0; i < request_queue_length; i++)
            if (now.request_queue[i] != MPI_REQUEST_NULL)
                nrequests++;
        _append_stat(stats, "waiting_requests", streamer->subgrid_worker,
                     nrequests, 1);

        // Send to statsd server
        config_send_statsd(streamer->work_cfg, stats);

        // Copy from "now" to "last", to use as reference next time
        memcpy(&last, &now, sizeof(struct streamer));
        memcpy(writers_last, writers_now, writers_size);
        last.writer = writers_last;

        // Determine when to next send stats, sleep
        while (next_stats <= get_time_ns()) {
            next_stats += sample_rate;
        }
        usleep((useconds_t) (1000000 * (next_stats - get_time_ns())) );
    }

    return NULL;
}

bool streamer_init(struct streamer *streamer,
                   struct work_config *wcfg, int subgrid_worker, int *producer_ranks)
{

    struct recombine2d_config *cfg = &wcfg->recombine;
    const int facets = wcfg->facet_workers * wcfg->facet_max_work;

    streamer->work_cfg = wcfg;
    streamer->subgrid_worker = subgrid_worker;
    streamer->producer_ranks = producer_ranks;

    streamer->have_kern = false;
    streamer->num_workers = omp_get_max_threads();
    streamer->wait_time = streamer->wait_in_time = streamer->degrid_time =
        streamer->task_start_time = streamer->recombine_time = 0;
    streamer->received_data = 0;
    streamer->received_subgrids = streamer->baselines_covered = 0;
    streamer->square_error_samples = 0;
    streamer->square_error_sum = 0;
    streamer->worst_error = 0;
    streamer->degrid_flops = 0;
    streamer->produced_chunks = 0;
    streamer->task_yields = 0;
    streamer->subgrid_tasks = 0;
    streamer->finished = false;

    // Load gridding kernel
    if (wcfg->gridder_path) {
        if (load_sep_kern(wcfg->gridder_path, &streamer->kern))
            return false;

        // Reduce oversampling if requested
        if (wcfg->vis_gridder_downsample) {
            const int downsample = wcfg->vis_gridder_downsample;
            streamer->kern.oversampling /= downsample;
            int i;
            for (i = 1; i < streamer->kern.oversampling; i++) {
                memcpy(streamer->kern.data + i * streamer->kern.stride,
                       streamer->kern.data + i * downsample * streamer->kern.stride,
                       sizeof(double) * streamer->kern.size);
            }
        }

        streamer->have_kern = true;
    }

    // Calculate size of queues
    streamer->queue_length = wcfg->vis_subgrid_queue_length;
    streamer->vis_queue_length = wcfg->vis_chunk_queue_length;
    streamer->writer_count = (wcfg->vis_path ? wcfg->vis_writer_count : 0);
    hbool_t hdf5_threadsafe;
    H5is_library_threadsafe(&hdf5_threadsafe);
    if (!wcfg->vis_fork_writer && streamer->writer_count > 1 && !hdf5_threadsafe) {
        fprintf(stderr, "WARNING: libhdf5 is not thread safe, using only one writer thread!\n");
        streamer->writer_count = 1;
    }
    if (streamer->writer_count > 0) {
        streamer->vis_queue_per_writer = streamer->vis_queue_length / streamer->writer_count;
    } else {
        streamer->vis_queue_per_writer = 0;
    }

    const int nmbf_length = cfg->NMBF_NMBF_size / sizeof(double complex);
    const size_t queue_size = (size_t)sizeof(double complex) * nmbf_length * facets * streamer->queue_length;
    const size_t sg_queue_size = (size_t)cfg->SG_size * streamer->queue_length;
    const size_t requests_size = (size_t)sizeof(MPI_Request) * facets * streamer->queue_length;
    struct vis_spec *const spec = &streamer->work_cfg->spec;
    const int vis_data_size = sizeof(double complex) * spec->time_chunk * spec->freq_chunk;
    printf("Allocating %.3g GB subgrid queue, %.3g GB visibility queue\n",
           (double)(queue_size+sg_queue_size+requests_size) / 1e9,
           (double)((size_t)streamer->vis_queue_length * (vis_data_size + 6 * sizeof(int))) / 1e9);

    // Allocate receive queue
    streamer->nmbf_queue = (double complex *)malloc(queue_size);
    streamer->request_queue = (MPI_Request *)malloc(requests_size);
    streamer->request_work = (int *)malloc(sizeof(int) * streamer->queue_length);
    streamer->subgrid_queue = (double complex *)malloc(sg_queue_size);
    streamer->subgrid_locks = (int *)calloc(sizeof(int), streamer->queue_length);
    streamer->skip_receive = (bool *)calloc(sizeof(bool), wcfg->subgrid_max_work);
    if (!streamer->nmbf_queue || !streamer->request_queue ||
        !streamer->subgrid_queue || !streamer->subgrid_locks ||
        !streamer->skip_receive) {

        fprintf(stderr, "ERROR: Could not allocate subgrid queue!\n");
        return false;
    }

    // Populate receive queue
    int iwork;
    for (iwork = 0; iwork < wcfg->subgrid_max_work && iwork < streamer->queue_length; iwork++) {
        streamer_ireceive(streamer, iwork, iwork);
    }
    for (; iwork < streamer->queue_length; iwork++) {
        int i;
        for (i = 0; i < facets; i++) {
            *request_slot(streamer, iwork, i) = MPI_REQUEST_NULL;
        }
        streamer->request_work[iwork] = -1;
    }

    // Plan FFTs
    streamer->subgrid_plan = fftw_plan_dft_2d(cfg->xM_size, cfg->xM_size,
                                              streamer->subgrid_queue,
                                              streamer->subgrid_queue,
                                              FFTW_BACKWARD, FFTW_MEASURE);
    // Allocate visibility queue
    streamer->vis_queue_size = (size_t)streamer->vis_queue_length * vis_data_size;
    streamer->vis_chunks_size = (size_t)streamer->vis_queue_length * sizeof(struct streamer_chunk);
    if (wcfg->vis_fork_writer) {
        streamer->vis_queue = mmap(NULL, streamer->vis_queue_size,
                                   PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        streamer->vis_chunks = mmap(NULL, streamer->vis_chunks_size,
                                    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    } else {
        streamer->vis_queue = malloc((size_t)streamer->vis_queue_length * vis_data_size);
        streamer->vis_chunks = malloc((size_t)streamer->vis_queue_length * sizeof(struct streamer_chunk));
    }
    if (!streamer->vis_queue || !streamer->vis_chunks) {

        fprintf(stderr, "ERROR: Could not allocate visibility queue!\n");
        return false;
    }

    // Initialise writer thread data
    streamer->writer = NULL;
    streamer->writer_size = streamer->writer_count * sizeof(struct streamer_writer);
    if (streamer->writer_count > 0) {
        if (wcfg->vis_fork_writer) {
            printf("Using %d writer processes\n", streamer->writer_count);
            streamer->writer = mmap(NULL, streamer->writer_size,
                                    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        } else {
            printf("Using %d writer threads\n", streamer->writer_count);
            streamer->writer = calloc(1, streamer->writer_size);
        }
        int i;
        for (i = 0; i < streamer->writer_count; i++) {
            struct streamer_writer *writer = streamer->writer + i;
            writer->subgrid_worker = subgrid_worker;
            writer->index = subgrid_worker * streamer->writer_count + i;
            writer->work_cfg = wcfg;
            writer->file = writer->group = -1;
            writer->queue_length = streamer->vis_queue_per_writer;
            writer->in_ptr = writer->out_ptr =
                writer->to_write = 0;
            writer->queue = streamer->vis_chunks + i * streamer->vis_queue_per_writer;
            int j;
            for (j = 0; j < streamer->vis_queue_per_writer; j++) {
                sem_init(&writer->queue[j].in_lock, wcfg->vis_fork_writer, 1);
                sem_init(&writer->queue[j].out_lock, wcfg->vis_fork_writer, 0);
                writer->queue[j].vis = streamer->vis_queue +
                    spec->time_chunk * spec->freq_chunk * (i * streamer->vis_queue_per_writer + j);
            }

            // Now either fork the writer or start a thread
            if (wcfg->vis_fork_writer) {
                writer->pid = fork();
                if (!writer->pid) {
                    streamer_writer_thread(writer);
                    exit(0);
                }
            } else {
                pthread_create(&writer->thread, NULL, streamer_writer_thread, writer);
            }
        }
    }

    return true;
}

void streamer_free(struct streamer *streamer,
                   double stream_start)
{

    // Wait for writer to actually finish
    int i;
    if (streamer->writer_count > 0) {
        printf("Finishing writes...\n");
        for (i = 0; i < streamer->writer_count; i++) {
            struct streamer_writer *writer = streamer->writer + i;
            if (streamer->work_cfg->vis_fork_writer) {
                waitpid(writer->pid, NULL, 0);
            } else {
                pthread_join(writer->thread, NULL);
            }
        }
    }

    double stream_time = get_time_ns() - stream_start;
    printf("Streamed for %.2fs\n", stream_time);
    printf("Received %.2f GB (%ld subgrids, %ld baselines)\n",
           (double)streamer->received_data / 1000000000, streamer->received_subgrids,
           streamer->baselines_covered);
    printf("Receiver: Wait: %gs, Recombine: %gs, Idle: %gs\n",
           streamer->wait_time, streamer->recombine_time,
           stream_time - streamer->wait_time - streamer->recombine_time);
    printf("Worker: Wait: %gs, Degrid: %gs, Idle: %gs\n",
           streamer->wait_in_time,
           streamer->degrid_time,
           streamer->num_workers * stream_time
           - streamer->wait_in_time - streamer->degrid_time
           - streamer->wait_time - streamer->recombine_time);
    printf("Operations: degrid %.1f GFLOP/s (%ld chunks)\n",
           (double)streamer->degrid_flops / stream_time / 1000000000,
           streamer->produced_chunks);
    if (streamer->square_error_samples > 0) {
        // Calculate root mean square error
        double rmse = sqrt(streamer->square_error_sum / streamer->square_error_samples);
        // Normalise by assuming that the energy of sources is
        // distributed evenly to all grid points
        const int source_count = streamer->work_cfg->produce_source_count;
        const int image_size = streamer->work_cfg->recombine.image_size;
        double energy = (double)source_count / image_size / image_size;
        printf("Accuracy: RMSE %g, worst %g (%ld samples)\n", rmse / energy,
               streamer->worst_error / energy, streamer->square_error_samples);
    }

    for (i = 0; i < streamer->writer_count; i++) {
        struct streamer_writer *writer = streamer->writer + i;

        int j;
        for (j = 0; j < streamer->vis_queue_per_writer; j++) {
            sem_destroy(&writer->queue[j].in_lock);
            sem_destroy(&writer->queue[j].out_lock);
        }

        // Then print stats. The above join can hang a bit as data
        // gets flushed, so re-determine stream time.
        printf("Writer %d: %.2f GB (rewritten %.2f GB), rate %.2f GB/s (%.2f GB/s effective)\n",
               writer->index,
               (double)writer->written_vis_data / 1000000000,
               (double)writer->rewritten_vis_data / 1000000000,
               (double)writer->written_vis_data / 1000000000 / stream_time,
               (double)(writer->written_vis_data - writer->rewritten_vis_data)
               / 1000000000 / stream_time);
        printf("Writer %d: Wait: %gs, Read: %gs, Write: %gs, Idle: %gs\n", writer->index,
               writer->wait_out_time, writer->read_time, writer->write_time,
               stream_time - writer->wait_out_time - writer->read_time - writer->write_time);
    }

    free(streamer->nmbf_queue); free(streamer->subgrid_queue);
    free(streamer->request_queue); free(streamer->subgrid_locks);
    free(streamer->skip_receive);
    fftw_free(streamer->subgrid_plan);
    if (streamer->work_cfg->vis_fork_writer) {
        munmap(streamer->vis_queue, streamer->vis_queue_size);
        munmap(streamer->vis_chunks, streamer->vis_chunks_size);
        if (streamer->writer) {
            munmap(streamer->writer, streamer->writer_size);
        }
    } else {
        free(streamer->vis_queue); free(streamer->vis_chunks);
        free(streamer->writer);
    }

}

void streamer(struct work_config *wcfg, int subgrid_worker, int *producer_ranks)
{

    struct streamer streamer;
    if (!streamer_init(&streamer, wcfg, subgrid_worker, producer_ranks)) {
        return;
    }

    double stream_start = get_time_ns();

    // Add reader, writer and statistic threads to OpenMP
    // threads. Those will idle most of the time, and therefore should
    // not count towards worker limit.
    int num_threads = streamer.num_workers;
    num_threads++; // reader thread
    if (wcfg->statsd_socket >= 0) {
        num_threads++; // statistics thread
    }
    omp_set_num_threads(num_threads);
    printf("Waiting for data (%d threads)...\n", num_threads);

    // Start doing work. Note that all the actual work will be added
    // by the reader thread, which will generate OpenMP tasks to be
    // executed by the following parallel section. All we're doing
    // here is yielding to them and shutting everything down once all
    // work has been completed.
#pragma omp parallel sections
    {
#pragma omp section
    {
        streamer_reader_thread(&streamer);

        // Wait for tasks to finish (could do a taskwait, but that
        // might mess up the thread balance)
        while (streamer.subgrid_tasks > 0) {
            usleep(10000);
        }
        #pragma omp taskwait // Just to be sure

        // All work is done - signal writer tasks to exit
        streamer.finished = true;
        int writer = 0;
        for (writer = 0; writer < streamer.writer_count; writer++) {
            struct streamer_chunk *slot = writer_push_slot(streamer.writer + writer, 0, 0, -1, -1);
            sem_post(&slot->out_lock);
        }
    }
#pragma omp section
    {
        if (wcfg->statsd_socket >= 0) {
            streamer_publish_stats(&streamer);
        }
    }

    } // #pragma omp parallel sections

    streamer_free(&streamer, stream_start);

}
