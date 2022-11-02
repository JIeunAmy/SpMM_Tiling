# data download
### ./download_small.sh

# compile
### nvcc matmul_test.cu -o matmul_test

# run
## ./matmul_test {data_dir or data_file(.mtx)} {funtion_type} {funtion_type_int} {targeted_execution} {TI} {TJ}
* select data_dir or datafile. if want to run specific file write file name and set {targeted_execution} as 1. If set targeted_execution as 1, must write TI, TJ.
* function type is either shared or non_shared
* if you choose kernel that doesn't use shared memory, set funtion_type_int as 0. otherwise, set function_type_int as 1.
 
## example
### run with data diretory, shared memory using kernel, full tiling set testing
./matmul_test {data_dir} shared 1 0

### run with data file, non shared memory using kernel, tiling with specific tile parameter with TI = 16,TJ = 64
./matmul_test {data_file} non_shared 0 1 16 64
