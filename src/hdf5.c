
#include "grid.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include <hdf5.h>
#include <stdarg.h>
#include <fcntl.h>
#include <unistd.h>

// Complex data type
hid_t dtype_cpx;

void init_dtype_cpx() {

    // HDF5 has no native complex datatype, so we mirror h5py here and
    // declare a compound equivalent.
    dtype_cpx = H5Tcreate(H5T_COMPOUND, sizeof(double complex));
    H5Tinsert(dtype_cpx, "r", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(dtype_cpx, "i", 8, H5T_NATIVE_DOUBLE);

}

bool load_ant_config(const char *filename, struct ant_config *cfg) {

    // Open file
    printf("Reading %s...\n", filename);
    hid_t cfg_f = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (cfg_f < 0) {
        fprintf(stderr, "Could not open antenna configuration file %s!\n", filename);
        return false;
    }
    hid_t cfg_g = H5Gopen(cfg_f, "cfg", H5P_DEFAULT);
    if (cfg_g < 0) {
        H5Fclose(cfg_f);
        fprintf(stderr, "Could not open 'cfg' group in antenna configuration file %s!\n", filename);
        return false;
    }

    // Read name
    hid_t name_a;
    if ((name_a = H5Aopen(cfg_g, "name", H5P_DEFAULT)) < 0 ||
        H5Tget_class(H5Aget_type(name_a)) != H5T_STRING ||
        H5Aread(name_a, H5Aget_type(name_a), &cfg->name) < 0) {

        H5Gclose(cfg_g);
        H5Fclose(cfg_f);
        fprintf(stderr, "Could not read 'name' attribute from antenna configuration file %s!\n", filename);
        return false;
    }

    // Read data, verify shape (... quite verbose ...)
    hid_t xyz_ds = H5Dopen(cfg_g, "xyz", H5P_DEFAULT);
    hsize_t xyz_dim[2];
    if (!(H5Tget_size(H5Dget_type(xyz_ds)) == sizeof(double) &&
          H5Sget_simple_extent_ndims(H5Dget_space(xyz_ds)) == 2 &&
          H5Sget_simple_extent_dims(H5Dget_space(xyz_ds), xyz_dim, NULL) >= 0 &&
          xyz_dim[1] == 3)) {

        H5Dclose(xyz_ds);
        H5Gclose(cfg_g);
        H5Fclose(cfg_f);
        fprintf(stderr, "Could not read 'xyz' data from antenna configuration file %s!\n", filename);
        return false;
    }

    // Set/read data
    cfg->ant_count = xyz_dim[0];
    cfg->xyz = (double *)malloc(sizeof(double) * 3 * cfg->ant_count);
    H5Dread(xyz_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cfg->xyz);

    // Done
    H5Dclose(xyz_ds);
    H5Gclose(cfg_g);
    H5Fclose(cfg_f);
    return true;
}

struct bl_stats {
    uint64_t vis_count, total_vis_count;
    double u_min, u_max;
    double v_min, v_max;
    double w_min, w_max;
    double t_min, t_max;
    double f_min, f_max;
};

static bool load_vis_group(hid_t vis_g, struct bl_data *bl,
                           int a1, int a2,
                           double min_len, double max_len,
                           struct bl_stats *stats) {

    // Read data, verify shape (... quite verbose ...)
    hid_t freq_ds = H5Dopen(vis_g, "frequency", H5P_DEFAULT);
    hid_t time_ds = H5Dopen(vis_g, "time", H5P_DEFAULT);
    hid_t uvw_ds = H5Dopen(vis_g, "uvw", H5P_DEFAULT);
    hid_t vis_ds = H5Dopen(vis_g, "vis", H5P_DEFAULT);
    hsize_t freq_dim, time_dim, uvw_dim[2], vis_dim[3];
    if (!(H5Sget_simple_extent_ndims(H5Dget_space(freq_ds)) == 1 &&
          H5Tget_size(H5Dget_type(freq_ds)) == sizeof(double) &&
          H5Sget_simple_extent_dims(H5Dget_space(freq_ds), &freq_dim, NULL) >= 0 &&
          H5Sget_simple_extent_ndims(H5Dget_space(time_ds)) == 1 &&
          H5Tget_size(H5Dget_type(time_ds)) == sizeof(double) &&
          H5Sget_simple_extent_dims(H5Dget_space(time_ds), &time_dim, NULL) >= 0 &&
          H5Sget_simple_extent_ndims(H5Dget_space(uvw_ds)) == 2 &&
          H5Tget_size(H5Dget_type(uvw_ds)) == sizeof(double) &&
          H5Sget_simple_extent_dims(H5Dget_space(uvw_ds), uvw_dim, NULL) >= 0 &&
          uvw_dim[0] == time_dim && uvw_dim[1] == 3 &&
          H5Sget_simple_extent_ndims(H5Dget_space(vis_ds)) == 3 &&
          H5Tget_size(H5Dget_type(vis_ds)) == sizeof(double complex) &&
          H5Sget_simple_extent_dims(H5Dget_space(vis_ds), vis_dim, NULL) >= 0 &&
          vis_dim[0] == time_dim && vis_dim[1] == freq_dim && vis_dim[2] == 1)) {

        H5Dclose(freq_ds);
        H5Dclose(time_ds);
        H5Dclose(uvw_ds);
        H5Dclose(vis_ds);
        return false;
    }

    // Determine visibility count
    int vis_c = vis_dim[0] * vis_dim[1] * vis_dim[2];
    if (stats) { stats->total_vis_count += vis_c; }

    // Use first uvw to decide whether to skip baseline
    bl->uvw_m = (double *)malloc(uvw_dim[0] * uvw_dim[1] * sizeof(double));
    H5Dread(uvw_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->uvw_m);
    double len = sqrt(bl->uvw_m[0] * bl->uvw_m[0] +
                      bl->uvw_m[1] * bl->uvw_m[1]);
    if (len < min_len || len >= max_len) {
        free(bl->uvw_m);
        H5Dclose(freq_ds);
        H5Dclose(time_ds);
        H5Dclose(uvw_ds);
        H5Dclose(vis_ds);
        return false;
    }
    if (stats) { stats->vis_count += vis_c; }

    // Read the baseline
    bl->antenna1 = a1;
    bl->antenna2 = a2;
    bl->time_count = time_dim;
    bl->freq_count = freq_dim;
    bl->time = (double *)malloc(time_dim * sizeof(double));
    bl->freq = (double *)malloc(freq_dim * sizeof(double));
    bl->vis = (double complex *)malloc(vis_c * sizeof(double complex));
    H5Dread(time_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->time);
    H5Dread(freq_ds, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->freq);
    H5Dread(vis_ds, dtype_cpx, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->vis);

    // Close groups
    H5Dclose(freq_ds);
    H5Dclose(time_ds);
    H5Dclose(uvw_ds);
    H5Dclose(vis_ds);

    // Statistics
    /* bl->u_min = DBL_MAX; bl->u_max = -DBL_MAX; */
    /* bl->v_min = DBL_MAX; bl->v_max = -DBL_MAX; */
    /* bl->w_min = DBL_MAX; bl->w_max = -DBL_MAX; */
    /* bl->t_min = DBL_MAX; bl->t_max = -DBL_MAX; */
    /* bl->f_min = DBL_MAX; bl->f_max = -DBL_MAX; */
    /* int j; */
    /* for (j = 0; j < freq_dim; j++) { */
    /*     if (bl->f_min > bl->freq[j]) { bl->f_min = bl->freq[j]; } */
    /*     if (bl->f_max < bl->freq[j]) { bl->f_max = bl->freq[j]; } */
    /* } */
    /* for (j = 0; j < time_dim; j++) { */
    /*     if (bl->t_min > bl->time[j])    { bl->t_min = bl->time[j]; } */
    /*     if (bl->t_max < bl->time[j])    { bl->t_max = bl->time[j]; } */
    /*     if (bl->u_min > bl->uvw_m[3*j+0]) { bl->u_min = bl->uvw_m[3*j+0]; } */
    /*     if (bl->u_max < bl->uvw_m[3*j+0]) { bl->u_max = bl->uvw_m[3*j+0]; } */
    /*     if (bl->v_min > bl->uvw_m[3*j+1]) { bl->v_min = bl->uvw_m[3*j+1]; } */
    /*     if (bl->v_max < bl->uvw_m[3*j+1]) { bl->v_max = bl->uvw_m[3*j+1]; } */
    /*     if (bl->w_min > bl->uvw_m[3*j+2]) { bl->w_min = bl->uvw_m[3*j+2]; } */
    /*     if (bl->w_max < bl->uvw_m[3*j+2]) { bl->w_max = bl->uvw_m[3*j+2]; } */
    /* } */

    /* if (stats) { */
    /*     if (stats->f_min > bl->f_min) { stats->f_min = bl->f_min; } */
    /*     if (stats->f_max < bl->f_max) { stats->f_max = bl->f_max; } */
    /*     if (stats->t_min > bl->t_min) { stats->t_min = bl->t_min; } */
    /*     if (stats->t_max < bl->t_max) { stats->t_max = bl->t_max; } */
    /*     if (stats->u_min > bl->u_min) { stats->u_min = bl->u_min; } */
    /*     if (stats->u_max < bl->u_max) { stats->u_max = bl->u_max; } */
    /*     if (stats->v_min > bl->v_min) { stats->v_min = bl->v_min; } */
    /*     if (stats->v_max < bl->v_max) { stats->v_max = bl->v_max; } */
    /*     if (stats->w_min > bl->w_min) { stats->w_min = bl->w_min; } */
    /*     if (stats->w_max < bl->w_max) { stats->w_max = bl->w_max; } */
    /* } */

    return true;
}

int load_vis(const char *filename, struct vis_data *vis,
             double min_len, double max_len) {

    // Open file
    printf("Reading %s...\n", filename);
    hid_t vis_f = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (vis_f < 0) {
        fprintf(stderr, "Could not open visibility file %s!\n", filename);
        return 1;
    }
    hid_t vis_g = H5Gopen(vis_f, "vis", H5P_DEFAULT);
    if (vis_g < 0) {
        fprintf(stderr, "Could not open 'vis' group in visibility file %s!\n", filename);
        return 1;
    }

    // Set up statistics
    struct bl_stats stats;
    stats.vis_count = stats.total_vis_count = 0;
    stats.u_min = DBL_MAX; stats.u_max = -DBL_MAX;
    stats.v_min = DBL_MAX; stats.v_max = -DBL_MAX;
    stats.w_min = DBL_MAX; stats.w_max = -DBL_MAX;
    stats.t_min = DBL_MAX; stats.t_max = -DBL_MAX;
    stats.f_min = DBL_MAX; stats.f_max = -DBL_MAX;

    // Check whether "vis" a flat visibility group (legacy - should
    // not have made a data set in this format in the first place.)
    hid_t type_a; char *type_str;
    int bl = 0;
    if (H5Aexists(vis_g, "type") &&
        (type_a = H5Aopen(vis_g, "type", H5P_DEFAULT)) >= 0 &&
        H5Tget_class(H5Aget_type(type_a)) == H5T_STRING &&
        H5Aread(type_a, H5Aget_type(type_a), &type_str) >= 0 &&
        strcmp(type_str, "Visibility") == 0) {

        // Read visibilities
        struct bl_data data;
        if (!load_vis_group(vis_g, &data, 0, 0, -DBL_MAX, DBL_MAX, &stats)) {
            H5Gclose(vis_g);
            H5Fclose(vis_f);
            return 1;
        }

        // Read antenna datasets
        hid_t a1_ds = H5Dopen(vis_g, "antenna1", H5P_DEFAULT);
        hid_t a2_ds = H5Dopen(vis_g, "antenna2", H5P_DEFAULT);
        hsize_t a1_dim, a2_dim;
        if (!(H5Sget_simple_extent_ndims(H5Dget_space(a1_ds)) == 1 &&
              H5Tget_size(H5Dget_type(a1_ds)) == sizeof(int64_t) &&
              H5Sget_simple_extent_dims(H5Dget_space(a1_ds), &a1_dim, NULL) >= 0 &&
              a1_dim == data.time_count &&
              H5Sget_simple_extent_ndims(H5Dget_space(a2_ds)) == 1 &&
              H5Tget_size(H5Dget_type(a2_ds)) == sizeof(int64_t) &&
              H5Sget_simple_extent_dims(H5Dget_space(a2_ds), &a2_dim, NULL) >= 0 &&
              a2_dim == data.time_count)) {

            free(data.uvw_m);
            free(data.time);
            free(data.freq);
            free(data.vis);
            H5Gclose(vis_g);
            H5Fclose(vis_f);
            return 1;

        }

        // Read antenna arrays
        int64_t *a1 = (int64_t *)malloc(a1_dim * sizeof(int64_t));
        int64_t *a2 = (int64_t *)malloc(a1_dim * sizeof(int64_t));
        H5Dread(a1_ds, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a1);
        H5Dread(a2_ds, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a2);
        H5Dclose(a1_ds);
        H5Dclose(a2_ds);

        // Split by baseline. We assume every visibility needs its own baseline.
        vis->bl_count = data.time_count;
        vis->bl = (struct bl_data *)calloc(vis->bl_count, sizeof(struct bl_data));
        stats.vis_count = 0;
        int i;
        for (i = 0; i < data.time_count; i++) {

            // Calculate baseline length (same check as in load_vis_group)
            double len = sqrt(data.uvw_m[3*i+0] * data.uvw_m[3*i+0] +
                              data.uvw_m[3*i+1] * data.uvw_m[3*i+1]);
            if (len < min_len || len >= max_len) {
                printf("asd %g\n", len);
                continue;
            }

            // Create 1-visibility baseline
            vis->bl[bl].antenna1 = a1[i];
            vis->bl[bl].antenna2 = a2[i];
            vis->bl[bl].time_count = 1;
            vis->bl[bl].freq_count = data.freq_count;
            vis->bl[bl].uvw_m = (double *)malloc(3 * sizeof(double));
            vis->bl[bl].time = (double *)malloc(sizeof(double));
            vis->bl[bl].freq = (double *)malloc(data.freq_count * sizeof(double));
            vis->bl[bl].vis = (double complex *)malloc(data.freq_count * sizeof(double complex));
            vis->bl[bl].time[0] = data.time[i];
            vis->bl[bl].uvw_m[0] = data.uvw_m[i*3+0];
            vis->bl[bl].uvw_m[1] = data.uvw_m[i*3+1];
            vis->bl[bl].uvw_m[2] = data.uvw_m[i*3+2];
            /* vis->bl[bl].u_min = vis->bl[bl].u_max = vis->bl[bl].uvw_m[0]; */
            /* vis->bl[bl].v_min = vis->bl[bl].v_max = vis->bl[bl].uvw_m[1]; */
            /* vis->bl[bl].w_min = vis->bl[bl].w_max = vis->bl[bl].uvw_m[2]; */
            /* vis->bl[bl].f_min = data.f_min; */
            /* vis->bl[bl].f_max = data.f_max; */
            int j;
            for (j = 0; j < data.freq_count; j++) {
                vis->bl[bl].freq[j] = data.freq[j];
                vis->bl[bl].vis[j] = data.vis[i*data.freq_count+j];
            }

            if (a1[i] > vis->antenna_count) { vis->antenna_count = a1[i]; }
            if (a2[i] > vis->antenna_count) { vis->antenna_count = a2[i]; }
            stats.vis_count++;
            bl++;
        }

        // Finish
        free(data.uvw_m);
        free(data.time);
        free(data.freq);
        free(data.vis);
        vis->bl_count = bl;

    } else {

        // Read number of baselines
        hsize_t nobjs = 0;
        H5Gget_num_objs(vis_g, &nobjs);
        vis->antenna_count = nobjs+1;
        if (vis->antenna_count == 0) {
            fprintf(stderr, "Found no antenna data in visibility file %s!\n", filename);
            H5Gclose(vis_g);
            H5Fclose(vis_f);
            return 1;
        }
        vis->bl_count = vis->antenna_count * (vis->antenna_count - 1) / 2;

        // Read baselines
        vis->bl = (struct bl_data *)calloc(vis->bl_count, sizeof(struct bl_data));
        int a1, bl = 0;
        for (a1 = 0; a1 < vis->antenna_count-1; a1++) {
            char a1_name[64];
            sprintf(a1_name, "%d", a1);
            hid_t a1_g = H5Gopen(vis_g, a1_name, H5P_DEFAULT);
            if (a1_g < 0) {
                fprintf(stderr, "Antenna1 %s not found!", a1_name);
                continue;
            }

            int a2;
            for (a2 = a1+1; a2 < vis->antenna_count; a2++) {
                char a2_name[64];
                sprintf(a2_name, "%d", a2);
                hid_t a2_g = H5Gopen(a1_g, a2_name, H5P_DEFAULT);
                if (a2_g < 0) {
                    fprintf(stderr, "Antenna2 %s/%s not found!", a1_name, a2_name);
                    continue;
                }

                // Read group data
                if (load_vis_group(a2_g, &vis->bl[bl], a1, a2, min_len, max_len, &stats)) {

                    // Next baseline!
                    bl++;
                }

                H5Gclose(a2_g);
            }
            H5Gclose(a1_g);
        }
        vis->bl_count = bl;
    }

    H5Gclose(vis_g);
    H5Fclose(vis_f);

    printf("\n");
    if (stats.vis_count < stats.total_vis_count) {
        printf("Have %d baselines, %"PRIu64" uvw positions, %"PRIu64" visibilities\n", vis->bl_count, stats.vis_count, stats.total_vis_count);
    } else {
        printf("Have %d baselines and %"PRIu64" visibilities\n", vis->bl_count, stats.vis_count);
    }
    printf("u range:     %.2f - %.2f lambda\n", stats.u_min*stats.f_max/c, stats.u_max*stats.f_max/c);
    printf("v range:     %.2f - %.2f lambda\n", stats.v_min*stats.f_max/c, stats.v_max*stats.f_max/c);
    printf("w range:     %.2f - %.2f lambda\n", stats.w_min*stats.f_max/c, stats.w_max*stats.f_max/c);
    printf("Antennas:    %d - %d\n"           , 0, vis->antenna_count);
    printf("t range:     %.6f - %.6f MJD UTC\n", stats.t_min, stats.t_max);
    printf("f range:     %.2f - %.2f MHz\n"    , stats.f_min/1e6, stats.f_max/1e6);

    return 0;
}

bool create_vis_group(hid_t vis_g, int freq_chunk, int time_chunk, bool skip_metadata,
                      struct bl_data *bl) {

    // Create a visibility group from baseline data *without* actually
    // writing any visibility data (data in "bl" will be ignored).
    // Instead, we will write a chunked visibility dataset with zeros
    // for fill value.

    // Create properties for compact and contigous data. Yes, it is
    // worth sharing them.
    static hid_t compact_ds_prop, cont_ds_prop, chunked_ds_prop;
    static bool ds_created = false;
    if (!ds_created) {
        compact_ds_prop = H5Pcreate(H5P_DATASET_CREATE);
        cont_ds_prop = H5Pcreate(H5P_DATASET_CREATE);
        chunked_ds_prop = H5Pcreate(H5P_DATASET_CREATE);
        ds_created = true;
        H5Pset_layout(compact_ds_prop, H5D_COMPACT);
        H5Pset_layout(cont_ds_prop, H5D_CONTIGUOUS);
        H5Pset_layout(chunked_ds_prop, H5D_CHUNKED);
        complex double fill_value = 0;
        H5Pset_fill_value(chunked_ds_prop, dtype_cpx, &fill_value);
    }
    hsize_t chunks[3] = { time_chunk, freq_chunk, 1 };
    H5Pset_chunk(chunked_ds_prop, 3, chunks);
    // Create datasets
    if (!skip_metadata) {
        hsize_t freq_size = bl->freq_count;
        hid_t freq_dsp = H5Screate_simple(1, &freq_size, NULL);
        hid_t freq_ds = H5Dcreate(vis_g, "frequency", H5T_IEEE_F64LE,
                                  freq_dsp, H5P_DEFAULT, compact_ds_prop, H5P_DEFAULT);
        if (freq_ds < 0) {
            fprintf(stderr, "failed to create frequency dataset!");
            H5Sclose(freq_dsp);
            return false;
        }
        H5Dwrite(freq_ds, H5T_NATIVE_DOUBLE, freq_dsp, freq_dsp, H5P_DEFAULT, bl->freq);
        H5Sclose(freq_dsp); H5Dclose(freq_ds);
        
        hsize_t time_size = bl->time_count;
        hid_t time_dsp = H5Screate_simple(1, &time_size, NULL);
        hid_t time_ds = H5Dcreate(vis_g, "time", H5T_IEEE_F64LE,
                                  time_dsp, H5P_DEFAULT, compact_ds_prop, H5P_DEFAULT);
        if (time_ds < 0) {
            fprintf(stderr, "failed to create time dataset!");
            H5Sclose(time_dsp);
            return false;
        }
        H5Dwrite(time_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->time);
        H5Sclose(time_dsp); H5Dclose(time_ds);
        
        hsize_t uvw_dims[2] = { bl->time_count, 3 };
        hid_t uvw_dsp = H5Screate_simple(2, uvw_dims, NULL);
        hid_t uvw_ds = H5Dcreate(vis_g, "uvw", H5T_IEEE_F64LE,
                                 uvw_dsp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (uvw_ds < 0) {
            fprintf(stderr, "failed to create coordinates dataset!");
            H5Sclose(uvw_dsp); H5Dclose(uvw_ds);
            return false;
        }
        H5Dwrite(uvw_ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bl->uvw_m);
        H5Sclose(uvw_dsp); H5Dclose(uvw_ds);
    }

    hsize_t vis_dims[3] = { 0, 0, 1 };
    hsize_t max_vis_dims[3] = { bl->time_count, bl->freq_count, 1 };
    hid_t vis_dsp = H5Screate_simple(3, vis_dims, max_vis_dims);
    hid_t vis_ds = H5Dcreate(vis_g, "vis", dtype_cpx,
                             vis_dsp, H5P_DEFAULT, chunked_ds_prop, H5P_DEFAULT);
    if (vis_ds < 0) {
        fprintf(stderr, "failed to create visibility dataset!");
        H5Sclose(vis_dsp);
        return false;
    }
    H5Sclose(vis_dsp); H5Dclose(vis_ds);

    return true;
}

static bool _rw_vis_chunk(hid_t vis_group,
                          struct bl_data *bl,
                          int time_chunk_size, int freq_chunk_size,
                          int time_chunk_ix, int freq_chunk_ix,
                          bool write, double complex *buf)
{

    // Generate name
    char name[128];
    sprintf(name, "%d/%d/vis", bl->antenna1, bl->antenna2);

    // Open dataset
    hid_t vis_ds = H5Dopen2(vis_group, name, H5P_DEFAULT);
    if (vis_ds < 0) {
        fprintf(stderr, "ERROR: Could not access dataset %s!\n", name);
        return false;
    }

    // Create memory and file data spaces
    hsize_t chunk_dims[] = { time_chunk_size, freq_chunk_size, 1 };
    hid_t chunk_dsp = H5Screate_simple(3, chunk_dims, chunk_dims);
    hsize_t vis_dims[] = { bl->time_count, bl->freq_count, 1 };
    hid_t vis_dsp = H5Screate_simple(3, vis_dims, vis_dims);

    // Select chunk (as one "block")
    hsize_t start[] = { time_chunk_ix * time_chunk_size,
                        freq_chunk_ix * freq_chunk_size, 0 };
    hsize_t stride[] = { 1,1,1 };
    assert(H5Sselect_hyperslab(vis_dsp, H5S_SELECT_SET, start, stride, stride, chunk_dims) >= 0);

    // Read or write chunk
    bool success = false;
    if (write) {
        success = H5Dwrite(vis_ds, dtype_cpx, chunk_dsp, vis_dsp, H5P_DEFAULT, buf) >= 0;
        assert(success);
    } else {
        success = H5Dread(vis_ds, dtype_cpx, chunk_dsp, vis_dsp, H5P_DEFAULT, buf) >= 0;
        assert(success);
    }

    H5Sclose(chunk_dsp);
    H5Sclose(vis_dsp);
    H5Dclose(vis_ds);
    return success;
}

bool read_vis_chunk(hid_t vis_group,
                    struct bl_data *bl,
                    int time_chunk_size, int freq_chunk_size,
                    int time_chunk_ix, int freq_chunk_ix,
                    double complex *buf)
{
    return _rw_vis_chunk(vis_group, bl, time_chunk_size, freq_chunk_size, time_chunk_ix, freq_chunk_ix,
                         false, buf);
}

bool write_vis_chunk(hid_t vis_group,
                    struct bl_data *bl,
                    int time_chunk_size, int freq_chunk_size,
                    int time_chunk_ix, int freq_chunk_ix,
                    double complex *buf)
{
    return _rw_vis_chunk(vis_group, bl, time_chunk_size, freq_chunk_size, time_chunk_ix, freq_chunk_ix,
                         true, buf);
}

int load_sep_kern(const char *filename, struct sep_kernel_data *sepkern, bool load_corr)
{

    // Open file
    hid_t sepkern_f = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (sepkern_f < 0) {
        fprintf(stderr, "Could not open separable kernel file %s!\n", filename);
        return 1;
    }

    // Open the data set
    hid_t dset = H5Dopen(sepkern_f, "sepkern/kern", H5P_DEFAULT);
    if (dset < 0) {
        fprintf(stderr, "'sepkern/kern' dataset could not be opened from file %s!\n", filename);
        H5Fclose(sepkern_f);
        return 1;
    }

    // Check that it has the expected format
    hsize_t dims[4];
    if (H5Sget_simple_extent_ndims(H5Dget_space(dset)) != 2 ||
        H5Tget_size(H5Dget_type(dset)) != sizeof(double) ||
        H5Sget_simple_extent_dims(H5Dget_space(dset), dims, NULL) < 0) {

        fprintf(stderr, "'sepkern/kern' dataset has wrong format in file %s!\n", filename);
        H5Dclose(dset);
        H5Fclose(sepkern_f);
        return 1;
    }

    // Allocate kernel memory, taking alignment into account
    const int align = 32; // For AVX2
    sepkern->oversampling = dims[0];
    sepkern->size = dims[1];
    sepkern->stride = align * ((sepkern->size * sizeof(double) + align - 1)  / align) / sizeof(double);
    hsize_t total_size = sepkern->oversampling * sepkern->stride;
    posix_memalign((void *)&sepkern->data, align, sizeof(double) * total_size);

    // Create data space that reflects kernel's memory layout, select
    // the bits we are actually going to write (i.e. tell it to ignore
    // the superflous bytes that make up the stride)
    hsize_t kdims[2] = { sepkern->oversampling, sepkern->stride };
    hid_t kernel_memspace = H5Screate_simple(2, kdims, kdims);
    hsize_t kstart[2] = { 0, 0 }; hsize_t kstride[2] = { 1, 1 };
    H5Sselect_hyperslab(kernel_memspace, H5S_SELECT_SET, kstart, kstride, kstride, dims);

    // Read kernel
    if (H5Dread(dset, H5T_NATIVE_DOUBLE, kernel_memspace, H5S_ALL, H5P_DEFAULT, sepkern->data) < 0) {
        fprintf(stderr, "Failed to read separable kernel data from %s!\n", filename);
        free(sepkern->data); sepkern->data = NULL;
        H5Sclose(kernel_memspace);
        H5Dclose(dset);
        H5Fclose(sepkern_f);
        return 1;
    }
    H5Sclose(kernel_memspace);
    H5Dclose(dset);

    // Read x0
    dset = H5Dopen(sepkern_f, "sepkern/x0", H5P_DEFAULT);
    if (dset < 0 ||
        H5Sget_simple_extent_ndims(H5Dget_space(dset)) != 0 ||
        H5Tget_size(H5Dget_type(dset)) != sizeof(double) ||
        H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &sepkern->x0) < 0 ) {

        if (!load_corr) {
            sepkern->x0 = 0.5;
        } else {
            fprintf(stderr, "Could not read 'sepkern/x0' from file %s!\n", filename);
            if (dset >= 0) H5Dclose(dset);
            free(sepkern->data); sepkern->data = NULL;
            H5Fclose(sepkern_f);
            return 1;
        }
    }
    if (dset >= 0) H5Dclose(dset);

    // Load correction if requested
    if (!load_corr) {
        sepkern->corr_size = 0;
        sepkern->corr = NULL;
        sepkern->x0 = 0.5;
    } else {

        // Open the data set
        hid_t dset = H5Dopen(sepkern_f, "sepkern/corr", H5P_DEFAULT);
        if (dset < 0) {
            fprintf(stderr, "'sepkern/corr' dataset could not be opened from file %s!\n", filename);
            free(sepkern->data); sepkern->data = NULL;
            H5Fclose(sepkern_f);
            return 1;
        }

        // Get size
        hsize_t dims[1];
        if (H5Sget_simple_extent_ndims(H5Dget_space(dset)) != 1 ||
            H5Tget_size(H5Dget_type(dset)) != sizeof(double) ||
            H5Sget_simple_extent_dims(H5Dget_space(dset), dims, NULL) < 0) {

            fprintf(stderr, "'sepkern/corr' dataset has wrong format in file %s!\n", filename);
            free(sepkern->data); sepkern->data = NULL;
            H5Dclose(dset);
            H5Fclose(sepkern_f);
            return 1;
        }

        // Read
        sepkern->corr_size = dims[0];
        sepkern->corr = malloc(sizeof(double) * dims[0]);
        if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, sepkern->corr) < 0) {
            fprintf(stderr, "Failed to read kernel correction data from %s!\n", filename);
            free(sepkern->data); sepkern->data = NULL;
            free(sepkern->corr); sepkern->corr = NULL;
            H5Dclose(dset);
            H5Fclose(sepkern_f);
            return 1;
        }
        H5Dclose(dset);

    }

    // Close file
    H5Fclose(sepkern_f);

    if(sepkern->corr) {
        printf("separable kernel: support %d (x%d oversampled), %d correction resolution, x0=%.2g\n",
               sepkern->size, sepkern->oversampling, sepkern->corr_size, sepkern->x0);
    } else {
        printf("separable kernel: support %d (x%d oversampled)\n",
               sepkern->size, sepkern->oversampling);
    }
    return 0;
}

// Quick routines for extracting single files (test support)

void *read_dump(int size, char *name, ...) {
    va_list ap;
    va_start(ap, name);
    char fname[256];
    vsnprintf(fname, 256, name, ap);
    int fd = open(fname, O_RDONLY, 0666);
    char *data = malloc(size);
    if (read(fd, data, size) != size) {
        fprintf(stderr, "failed to read enough data from %s!\n", fname);
        return 0;
    }
    close(fd);
    return data;
}

int write_dump(void *data, int size, char *name, ...) {
    va_list ap;
    va_start(ap, name);
    char fname[256];
    vsnprintf(fname, 256, name, ap);
    int fd = open(fname, O_CREAT | O_TRUNC | O_WRONLY, 0666);
    if (write(fd, data, size) != size) {
        fprintf(stderr, "failed to write data to %s!\n", fname);
        close(fd);
        return 1;
    }
    close(fd);
    return 0;
}

int get_npoints_hdf5(const char *file, char *name, ...)
{
    hid_t f = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
    va_list ap;
    va_start(ap, name);
    char dname[256];
    vsnprintf(dname, 256, name, ap);
    hid_t dset = H5Dopen(f, dname, H5P_DEFAULT);
    int npoints = H5Sget_simple_extent_npoints(H5Dget_space(dset));
    H5Dclose(dset); H5Fclose(f);
    return npoints;
}

void *read_hdf5(int size, const char *file, char *name, ...)
{
    hid_t f = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
    va_list ap;
    va_start(ap, name);
    char dname[256];
    vsnprintf(dname, 256, name, ap);
    hid_t dset = H5Dopen(f, dname, H5P_DEFAULT);
    // Check element size
    int elem_size = H5Tget_size(H5Dget_type(dset));
    // Check overall size
    int npoints = H5Sget_simple_extent_npoints(H5Dget_space(dset));
    if (npoints * elem_size != size) {
        fprintf(stderr, "Dataset %s in %s has wrong extend (%d*%d != %d)\n",
                dname, file, npoints, elem_size, size);
        H5Dclose(dset); H5Fclose(f);
        return NULL;
    }
    // Read data
    char *data = malloc(size);
    H5Dread(dset, H5Dget_type(dset), H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset); H5Fclose(f);
    return data;
}
