#!/bin/bash

# Quick example usage:
#
# $ sbatch -N 25 -n 125 ~/cam/imaging-iotest/scripts/iotest.slurm 8 256 256
#
# For 25 nodes, 25 facet workers, 100 stream workers, 8 threads per
# process, and 256x256 chunks (1 MB)

#SBATCH --time=00:30:00
#SBATCH -p skylake
#SBATCH -J sdp-iotest
#SBATCH -A SKA-SDP-SL2-CPU
#SBATCH --mail-type=FAIL
#SBATCH --no-requeue
#SBATCH --exclusive

#DW jobdw capacity=148TB access_mode=striped,private type=scratch
#lfs setstripe -S 1M -c -1 $DW_JOB_STRIPED

# module load gnu7 openmpi3 hdf5

# Read parameters
num_threads=$1
outdir=$DW_JOB_STRIPED
#outdir=/rds-d2/user/$USER/hpc-work/iotest
workdir=$HOME/cam/imaging-iotest/src

if [ -z "$2" ]; then
    freq_chunk=128
else
    freq_chunk=$2
fi
if [ -z "$3" ]; then
    time_chunk=128
else
    time_chunk=$3
fi

# Select image sizing by available (collective) memory
#options="--rec-set=16k-8k-512" # 4 GiB
#options="--rec-set=32k-8k-1k" # 16 GiB
#options="--rec-set=64k-16k-1k" # 64 GiB
#options="--rec-set=96k-12k-1k" # 144 GiB
options="--rec-set=128k-32k-2k" # 256 GiB, 25 facets
#options="--rec-set=256k-32k-2k" # 1 TiB, 81 facets

# Output full-resolution visibilities. Time and frequency ranges are
# given in "range/steps/chunks" format. Rule of thumb is that total
# amount of visibility data produced is
#
#    2MiB * time_count * freq_count
#
# So e.g. 512 time steps and 8192 frequency steps means ~ 8 TiB. Note
# that there are inefficiencies due to chunking, so actual size can be
# ~20% larger.
options+=" --vis-set=lowbd2 --time=-460:460/512/$time_chunk --freq=260e6:300e6/8192/$freq_chunk"

# Use exactly one or two facet workers per node (but can't have more
# facet workers than facets!). Rest of the workers will be streamer
# processes that actually write data.
facet_workers=$[1*${SLURM_JOB_NUM_NODES}]
options+=" --facet-workers=$facet_workers --send-queue=32 --subgrid-queue=128"

# Push statistics to statsd at localhost
#options+=" --statsd=127.0.0.1"

# Generate (and check) sources
options+=" --source-count=10"

# Set output file?
if [ ! -z "$outdir" ]; then
    options+=" $outdir/out%d.h5"
fi

#CMD="mpirun --tag-output --map-by node -npernode $mpi_tasks_per_node -np $np $application $options"
#MCA_PARAMS="--mca btl openib,self,vader  --mca btl_openib_if_include mlx5_0:1"
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
MAPPING_PARAMS="-ppn 1 -np $SLURM_NTASKS"

# Compose command line
CMD="mpirun -prepend-rank $MCA_PARAMS $MAPPING_PARAMS $workdir/iotest $options"

#! Pin processes to sockets
export I_MPI_PIN_DOMAIN=omp:compact
export OMP_NUM_THREADS=$num_threads
export OMP_PROC_BIND=true
export OMP_PLACES=sockets

# Make sure we have the right working directory
cd $workdir

echo -e "JobID: $SLURM_JOB_ID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "Output directory: $outdir"
echo -e "OMP_PROC_BIND=$OMP_PROC_BIND, OMP_PLACES=$OMP_PLACES"
echo -e "\nnumtasks=$SLURM_NTASKS, numnodes=$SLURM_JOB_NUM_NODES, OMP_NUM_THREADS=$OMP_NUM_THREADS"
df -h $outdir

# Clean up directory
if [ ! -z $outdir ]; then
    echo -e "\n==================\nCleaning $outdir...\n"
    if [ $SLURM_NODEID = 0 ]; then
        mkdir -p $outdir
        rm -f $outdir/out*.h5
    fi
    sleep 2
fi

echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

echo -e "\n=================="
echo "Finish time: `date`"

# Clean up afterwards
if [ ! -z "$outdir" -a $SLURM_NODEID = 0 ]; then
    echo -e "\n==================\nCleaning $outdir...\n"
    rm -f $outdir/out*.h5
fi
