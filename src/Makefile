
HDF5_INC ?= /usr/include/hdf5/serial
HDF5_LIB ?= /usr/lib/x86_64-linux-gnu/hdf5/serial

CFLAGS += -Wall -fopenmp -ffast-math -I$(HDF5_INC) -ggdb -march=native -O2
LDFLAGS += -fopenmp -ggdb -O2
LDLIBS = -L$(HDF5_LIB) -lm -lhdf5 -lfftw3
CC = mpicc
MPIRUN = mpirun

GRID_FILES = grid_avx2_16.c grid_avx2_12.c grid_avx2_8.c

IOTEST_OBJS = iotest.o recombine.o hdf5.o config.o producer.o streamer.o grid.o
TEST_RECOMBINE_OBJS = recombine.o test_recombine.o grid.o hdf5.o
TEST_CONFIG_OBJS = test_config.o config.o recombine.o  hdf5.o

iotest : $(IOTEST_OBJS)
test_recombine : $(TEST_RECOMBINE_OBJS)
test_config : $(TEST_CONFIG_OBJS)

grid_avx2_%.c : ../scripts/mk_grid_avx2.py
	python3 ../scripts/mk_grid_avx2.py $@
grid.o : $(GRID_FILES) grid.c
	$(CC) $(CFLAGS) grid.c -c -ogrid.o

.PHONY: test_T05 test_small
test_T05: iotest
	$(MPIRUN) -n 2 ./iotest --rec-set=T05 | grep RMSE
test_small: iotest
	OMP_NUM_THREADS=16 $(MPIRUN) -n 2 ./iotest --rec-set=small --vis-set=vlaa --subgrid-queue=128 --source-count=10 --freq=280e6:300e6/128/128 --grid=../data/grid/kernel_16_0.35.in | grep RMSE

.PHONY: clean
clean :
	rm -f $(IOTEST_OBJS) $(TEST_RECOMBINE_OBJS) $(TEST_CONFIG_OBJS) grid test_recombine recombine test_config
