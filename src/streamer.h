
#ifndef STREAMER_H
#define STREAMER_H

#ifndef NO_MPI
#include <mpi.h>
#else
#define MPI_Request int
#define MPI_REQUEST_NULL 0
#endif

#ifndef __APPLE__
#include <semaphore.h>
#else
#include <dispatch/dispatch.h>
#define sem_t dispatch_semaphore_t
#endif

#include "config.h"

struct streamer_chunk
{
    struct bl_data *bl_data;
    int tchunk, fchunk;
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

    struct sep_kernel_data *kern;

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

    // Transfer pattern for w-towers
    double complex *wtransfer;

    // Statistics
    int num_workers;
    double wait_time;
    double wait_in_time;
    double recombine_time, check_time, fft_time, task_start_time, degrid_time;
    uint64_t received_data, received_subgrids, baselines_covered;
    uint64_t vis_error_samples, grid_error_samples;
    double vis_error_sum, vis_worst_error, grid_error_sum, grid_worst_error;
    uint64_t degrid_flops;
    uint64_t produced_chunks;
    uint64_t task_yields;

    // Signal for being finished
    bool finished;
};

inline static double complex *nmbf_slot(struct streamer *streamer,
                                        int slot, int facet)
{
    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    const int xM_yN_size = streamer->work_cfg->recombine.xM_yN_size;
    assert(facet >= 0 && facet < (facets == 0 ? 1 : facets));
    return streamer->nmbf_queue + xM_yN_size * xM_yN_size * ((slot * facets) + facet);
}

inline static MPI_Request *request_slot(struct streamer *streamer,
                                        int slot, int facet)
{
    const int facets = streamer->work_cfg->facet_workers * streamer->work_cfg->facet_max_work;
    assert(facet >= 0 && facet < (facets == 0 ? 1 : facets));
    return streamer->request_queue + (slot * facets) + facet;
}

inline static double complex *subgrid_slot(struct streamer *streamer,
                                           int slot)
{
    const int xM_size = streamer->work_cfg->recombine.xM_size;
    return streamer->subgrid_queue + xM_size * xM_size * slot;
}

void streamer_work(struct streamer *streamer,
                   int subgrid_work,
                   double complex *nmbf);
struct streamer_chunk *writer_push_slot(struct streamer_writer *writer,
                                        struct bl_data *bl_data,
                                        int tchunk, int fchunk);

#endif
