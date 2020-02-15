
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

struct streamer_chunk *writer_push_slot(struct streamer_writer *writer,
                                        struct bl_data *bl_data,
                                        int tchunk, int fchunk)
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
#ifndef __APPLE__
    sem_wait(&chunk->in_lock);
#else
    dispatch_semaphore_wait(chunk->in_lock, DISPATCH_TIME_FOREVER);
#endif

    // Set slot data
    chunk->bl_data = bl_data;
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
        if (work[iw].iu == work[subgrid_work].iu &&
            work[iw].iv == work[subgrid_work].iv &&
            work[iw].iw == work[subgrid_work].iw)
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
        //printf("Receiving iu=%d iv=%d iw=%d tag=%d facet=%d\n",
        //       work[subgrid_work].iu, work[subgrid_work].iv, work[subgrid_work].iw, tag, facet_work);
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
        if (work[iw].iu == work[iwork].iu &&
            work[iw].iv == work[iwork].iv &&
            work[iw].iw == work[iwork].iw)
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
#ifndef __APPLE__
        sem_wait(&chunk->out_lock);
#else
        dispatch_semaphore_wait(chunk->out_lock, DISPATCH_TIME_FOREVER);
#endif
        writer->wait_out_time += get_time_ns() - start;

        start = get_time_ns();

        // Obtain baseline data
        if (chunk->tchunk == -1 && chunk->fchunk == -1)
            break; // Signal to end thread
        if (chunk->tchunk == -2 && chunk->fchunk == -2) {
            #pragma omp atomic
                writer->to_write -= 1;
#ifndef __APPLE__
            sem_post(&chunk->in_lock);
#else
            dispatch_semaphore_signal(chunk->in_lock);
#endif
            writer->out_ptr = (writer->out_ptr + 1) % writer->queue_length;
            continue; // Signal to ignore chunk
        }
        const int nant =  wcfg->spec.cfg->ant_count;
        struct bl_data *bl_data = chunk->bl_data;
        double complex *vis_data = chunk->vis;

        // Read visibility chunk. If it was not yet set, this will
        // just fill the buffer with zeroes.
        int chunk_index = ((bl_data->antenna2 * nant + bl_data->antenna1)
                           * time_chunk_count + chunk->tchunk) * freq_chunk_count + chunk->fchunk;
        if (wcfg->vis_check_existing || chunks_written[chunk_index]) {
            read_vis_chunk(writer->group, bl_data,
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
            write_vis_chunk(writer->group, bl_data,
                            spec->time_chunk, spec->freq_chunk,
                            chunk->tchunk, chunk->fchunk,
                            vis_data_h5);

            writer->written_vis_data += vis_data_size;
            if (chunks_written[chunk_index])
                writer->rewritten_vis_data += vis_data_size;
            chunks_written[chunk_index] = true;

        }

        // Release "in" lock to mark the slot free for writing
        #pragma omp atomic
            writer->to_write -= 1;
#ifndef __APPLE__
        sem_post(&chunk->in_lock);
#else
        dispatch_semaphore_signal(chunk->in_lock);
#endif
        writer->out_ptr = (writer->out_ptr + 1) % writer->queue_length;
        writer->write_time += get_time_ns() - start;

    }

    H5Gclose(writer->group); H5Fclose(writer->file);

    return NULL;
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
        _append_stat(PARS(vis_error_samples), 1 / sample_rate);
        _append_stat(PARS(grid_error_samples), 1 / sample_rate);
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
        double derror_sum = now.vis_error_sum - last.vis_error_sum;
        uint64_t dsamples = now.vis_error_samples - last.vis_error_samples;
        if (dsamples > 0) {
            _append_stat(stats, "visibility_samples", streamer->subgrid_worker,
                         dsamples, 1);
            _append_stat(stats, "visibility_rmse", streamer->subgrid_worker,
                         10 * log(sqrt(derror_sum / dsamples) / streamer->work_cfg->source_energy) / log(10), 1);
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

    streamer->num_workers = omp_get_max_threads();
    streamer->wait_time = streamer->wait_in_time = streamer->degrid_time =
        streamer->task_start_time = streamer->recombine_time = 0;
    streamer->received_data = 0;
    streamer->received_subgrids = streamer->baselines_covered = 0;
    streamer->vis_error_samples = 0;
    streamer->vis_error_sum = 0;
    streamer->vis_worst_error = 0;
    streamer->grid_error_samples = 0;
    streamer->grid_error_sum = 0;
    streamer->grid_worst_error = 0;
    streamer->degrid_flops = 0;
    streamer->produced_chunks = 0;
    streamer->task_yields = 0;
    streamer->subgrid_tasks = 0;
    streamer->finished = false;

    // Load gridding kernel
    if (wcfg->gridder.data) {
        streamer->kern = &wcfg->gridder;
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
#ifndef __APPLE__
                sem_init(&writer->queue[j].in_lock, wcfg->vis_fork_writer, 1);
                sem_init(&writer->queue[j].out_lock, wcfg->vis_fork_writer, 0);
#else
                writer->queue[j].in_lock = dispatch_semaphore_create(1);
                writer->queue[j].out_lock = dispatch_semaphore_create(0);
#endif
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

bool streamer_free(struct streamer *streamer,
                   double stream_start)
{
    bool success = true;

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
    printf("Received %.2f GB (%"PRIu64" subgrids, %"PRIu64" baselines)\n",
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
    printf("Operations: degrid %.1f GFLOP/s (%"PRIu64" chunks)\n",
           (double)streamer->degrid_flops / stream_time / 1000000000,
           streamer->produced_chunks);
    if (streamer->vis_error_samples > 0) {
        // Calculate root mean square error
        const double grid_rmse = sqrt(streamer->grid_error_sum / streamer->grid_error_samples);
        const double vis_rmse = sqrt(streamer->vis_error_sum / streamer->vis_error_samples);
        // Normalise by assuming that the energy of sources is
        // distributed evenly to all grid points
        const double source_energy = streamer->work_cfg->source_energy;
        printf("Grid accuracy: RMSE %g, worst %g (%"PRIu64" samples)\n",
               grid_rmse / source_energy, streamer->grid_worst_error / source_energy,
               streamer->grid_error_samples);
        printf("Vis accuracy: RMSE %g, worst %g (%"PRIu64" samples)\n",
               vis_rmse / source_energy, streamer->vis_worst_error / source_energy,
               streamer->vis_error_samples);
        // Check against error bounds
        if (fmax(streamer->grid_worst_error, streamer->vis_worst_error)
            > streamer->work_cfg->vis_max_error * source_energy) {
            printf("ERROR: Accuracy worse than RMSE threshold of %g!\n",
                   streamer->work_cfg->vis_max_error);
            success = false;
        }
    }

    for (i = 0; i < streamer->writer_count; i++) {
        struct streamer_writer *writer = streamer->writer + i;

        int j;
        for (j = 0; j < streamer->vis_queue_per_writer; j++) {
#ifndef __APPLE__
            sem_destroy(&writer->queue[j].in_lock);
            sem_destroy(&writer->queue[j].out_lock);
#else
            dispatch_release(writer->queue[i].in_lock);
            dispatch_release(writer->queue[i].out_lock);
#endif
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

    return success;
}

int streamer(struct work_config *wcfg, int subgrid_worker, int *producer_ranks)
{

    struct streamer streamer;
    if (!streamer_init(&streamer, wcfg, subgrid_worker, producer_ranks)) {
        return 1;
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
            struct streamer_chunk *slot = writer_push_slot(
                 streamer.writer + writer, NULL, -1, -1);
#ifndef __APPLE__
            sem_post(&slot->out_lock);
#else
            dispatch_semaphore_signal(slot->out_lock);
#endif
        }
    }
#pragma omp section
    {
        if (wcfg->statsd_socket >= 0) {
            streamer_publish_stats(&streamer);
        }
    }
    } // #pragma omp parallel sections


    // Finalise streamer. Check for successful completion.
    if (streamer_free(&streamer, stream_start)) {
        return 0;
    } else {
        return 1;
    }
}
