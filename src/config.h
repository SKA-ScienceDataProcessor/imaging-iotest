
#ifndef CONFIG_H
#define CONFIG_H

#include "recombine.h"
#include "grid.h"

// Specification of a visibility set
struct vis_spec
{
    double fov; // (true) field of view
    struct ant_config *cfg; // antennas
    double dec; // declination (radian)
    double time_start; int time_count; int time_chunk; double time_step; // hour angle (radian)
    double freq_start; int freq_count; int freq_chunk; double freq_step; // (Hz)
    // Cached hour angle / declination cosinus & sinus
    double *ha_sin, *ha_cos;
    double dec_sin, dec_cos;
};

inline static int spec_time_chunks(struct vis_spec *spec) {
    return (spec->time_count + spec->time_chunk - 1) / spec->time_chunk;
}
inline static int spec_freq_chunks(struct vis_spec *spec) {
    return (spec->freq_count + spec->freq_chunk - 1) / spec->freq_chunk;
}

void bl_bounding_box(struct bl_data *bl_data, bool negate,
                     int tstep0, int tstep1,
                     int fstep0, int fstep1,
                     double *uvw_l_min, double *uvw_l_max);

// Work to do on a facet
struct facet_work
{
    int il, im;
    int facet_off_l, facet_off_m;
    char *path, *hdf5; // random if not set
    bool set; // empty otherwise
};

// Work to do for a subgrid on a baseline
struct subgrid_work_bl
{
    int a1, a2; // Baseline antennas
    int chunks; // Number of (time,frequency) chunks overlapping
    double min_w; // Minimum touched w-level (for sorting)
    struct bl_data *bl_data;
    struct subgrid_work_bl *next;
};

// Work to do for a subgrid
struct subgrid_work
{
    int iu, iv, iw; // Column/row/plane number. Used for grouping, so must be consistent across work items!
    int subgrid_off_u, subgrid_off_v, subgrid_off_w; // Midpoint offset in grid coordinates
    int nbl; // Baselines in this work bin
    char *check_path, *check_fct_path,
         *check_degrid_path, *check_hdf5; // check data if set
    double check_threshold, check_fct_threshold,
           check_degrid_threshold; // at what discrepancy to fail
    struct subgrid_work_bl *bls; // Baselines
};

struct work_config {

    // Fundamental dimensions (uvw grid / cubes)
    double theta; // size of (padded) image in radians (1/uvstep)
    double wstep; // distance of w-planes
    int sg_step, sg_step_w; // effective subgrid cube size (step length as above)
    struct vis_spec spec; // Visibility specification
    struct bl_data *bl_data; // Baseline data (e.g. UVWs)
    char *vis_path; // Visibility file (pattern)
    struct sep_kernel_data gridder, w_gridder; // uv/w gridder

    // Worker configuration
    int facet_workers; // number of facet workers
    int facet_max_work; // work list length per worker
    int facet_count; // Number of facets
    struct facet_work *facet_work; // facet work list (2d array - worker x work)
    int subgrid_workers; // number of subgrid workers
    int subgrid_max_work; // work list length per worker
    struct subgrid_work *subgrid_work; // subgrid work list (2d array - worker x work)
    int iu_min, iu_max, iv_min, iv_max; // subgrid columns/rows

    // Recombination configuration
    struct recombine2d_config recombine;

    // Source configuration (if we are working from a "sky model")
    int source_count;
    double *source_xy; // Source image positions [produce_souce_count x xy]
    double *source_lmn; // Source sky positions [produce_souce_count x lmn]
    double *source_corr; // Grid correction at source positions
    double source_energy; // Mean energy sources add to every grid cell

    // Parameters
    int config_dump_baseline_bins;
    int config_dump_subgrid_work;
    int produce_parallel_cols;
    int produce_retain_bf;
    int produce_batch_rows;
    int produce_queue_length;
    int vis_skip_metadata;
    int vis_bls_per_task;
    int vis_subgrid_queue_length;
    int vis_task_queue_length;
    int vis_chunk_queue_length;
    int vis_writer_count;
    int vis_fork_writer;
    int vis_check_existing;
    int vis_checks, grid_checks;
    double vis_max_error;
    int vis_round_to_wplane;

    // Statsd connection
    int statsd_socket;
    double statsd_rate;
};

// Return size of total grid in wavelengths
inline static double config_lambda(const struct work_config *cfg) {
    return cfg->recombine.image_size / cfg->theta;
}
// Return size of a subgrid as fraction of the entire grid
inline static double config_xA(const struct work_config *cfg) {
    return (double)cfg->recombine.xA_size / cfg->recombine.image_size;
}

double get_time_ns();

void config_init(struct work_config *cfg);
bool config_set(struct work_config *cfg,
                int image_size, int subgrid_spacing,
                char *pswf_file,
                int yB_size, int yN_size, int yP_size,
                int xA_size, int xM_size, int xMxN_yP_size);
bool config_assign_work(struct work_config *cfg,
                        int facet_workers, int subgrid_workers);

void config_free(struct work_config *cfg);

void config_set_visibilities(struct work_config *cfg,
                             struct vis_spec *spec,
                             const char *vis_path);
bool config_set_degrid(struct work_config *cfg,
                       const char *gridder_path, double gridder_x0, int downsample);

bool config_set_statsd(struct work_config *cfg,
                       const char *node, const char *service);
void config_send_statsd(struct work_config *cfg, const char *stat);

void config_load_facets(struct work_config *cfg, const char *path_fmt, const char *hdf5);
void config_check_subgrids(struct work_config *cfg,
                           double threshold, double fct_threshold, double degrid_threshold,
                           const char *check_fmt, const char *check_fct_fmt,
                           const char *check_degrid_fmt, const char *hdf5);

void config_set_sources(struct work_config *cfg, int count, unsigned int seed);

void vis_spec_to_bl_data(struct bl_data *bl, struct vis_spec *spec,
                         int a1, int a2);
bool create_bl_groups(hid_t vis_group, struct work_config *work_cfg, int worker);

int make_subgrid_tag(struct work_config *wcfg,
                     int subgrid_worker_ix, int subgrid_work_ix,
                     int facet_worker_ix, int facet_work_ix);

int producer(struct work_config *wcfg, int facet_worker, int *streamer_ranks);
int streamer(struct work_config *wcfg, int subgrid_worker, int *producer_ranks);

#endif // CONFIG_H
