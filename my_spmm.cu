#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "mmio_highlevel.h"

#define N 128
#define TK 4
#define TI 2
#define TJ 4
#define MIN(a,b) ({a>b? b: a;})
# define INDEX_DATA_TYPE unsigned char
__global__ void testing(  MAT_VAL_TYPE *y_device){
    int row = threadIdx.y;
    int col = threadIdx.x;
    //printf("***************%d %d#############\n",row,col);
    y_device[row*N+col] = 20;
}

__global__ void my_tiling(int rowA, int colA, MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, MAT_VAL_TYPE *csrValA, MAT_VAL_TYPE *x_device, MAT_VAL_TYPE *y_device){
    int ti = blockIdx.y;
    int tj = blockIdx.x;
    __shared__ MAT_VAL_TYPE dns_tile[TJ][N];
    //__shared__ MAT_VAL_TYPE out_tile[TI][N];

    int row = ((ti*TI)+threadIdx.y);
    int col = (tj*TJ)+threadIdx.x;
    //printf("row: %d col: %d, thread x: %d, thread y: %d block x: %d, block y: %d\n",row,col,threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    //shared memory loading
    for(int j=0;j<N;++j){
        dns_tile[col][j] = x_device[col*N+j];
        //out_tile[row][j] = 0;
    }
    __syncthreads();
    
    if(row>=rowA) return;
    if(col>=colA) return;

    MAT_VAL_TYPE sum = 0;
    for(int i = csrRowPtrA[row];i<csrRowPtrA[row+1];++i){
        if(csrColIdxA[i]<col) continue;
        //if(csrColIdxA[i]>=col+TJ) break;
        if(csrColIdxA[i]>col) break;
        if(csrColIdxA[i] == col) {
            for(int j=0;j<N;++j){
                sum = csrValA[i]*(dns_tile[col][j]);
                //atomicAdd(&out_tile[row][j],sum);
                atomicAdd(&y_device[row*N+j],sum);
                //printf("row: %d col: %d, valA: %.0f, valD: %.0f, sum: %2.0f, thread y: %d, thread x: %d, block y: %d, block x: %d\n",row,col,csrValA[i],dns_tile[col][j],y_device[row*N+j],threadIdx.y, threadIdx.x, blockIdx.y, blockIdx.x);
            }
            break;
        }
    }
    __syncthreads();
    // for(int i=0;i<N;++i){
    //     atomicAdd(&y_device[row*colA+i],out_tile[row][i]);
    //     printf("valA: %f,valD: %f, sum: %f, row: %d col: %d, thread x: %d, thread y: %d block x: %d, block y: %d\n",csrValA[i],dns_tile[col][i],out_tile[row][i],row,col,threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    // }
    // __syncthreads();
}
__global__ void j_stream(int rowA, int colA, MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, MAT_VAL_TYPE *csrValA, MAT_VAL_TYPE *x_device, MAT_VAL_TYPE *y_device){
    //printf("dddddddddddddddd\n");
    int ti = blockIdx.x;
    int tk = blockIdx.y;
    //__shared__ MAT_VAL_TYPE out_tile[TI][TK];
    // __shared__ MAT_VAL_TYPE *sp_tile[TI][colA];
    __shared__ MAT_VAL_TYPE dns_tile[TK][TK];

    int row = TI*ti + threadIdx.y;
    int col = TK*tk + threadIdx.x;
    MAT_VAL_TYPE sum = 0.0;
    if(row>=rowA || col>= colA)
        return;
    //printf("row: %d, col: %d\n",row,col);
    for (int i=0;i<colA;i++){
        dns_tile[i][col] = x_device[i*N+col];
    }
    __syncthreads();
    for(int j = csrRowPtrA[row];j<csrRowPtrA[row+1];++j){
        int jj = csrColIdxA[j];
        //sp_tile[row][jj] = csrValA[j];
        //dns_tile[jj][col] = x[jj*N+col];
        //sum += sp_tile[row][jj]*dns_tile[jj][col];
        sum += csrValA[j]*dns_tile[jj][col];
    }
    //y_device[row*N+col] = sum;
    atomicAdd(&y_device[row*N+col],sum);
    //printf("#####################sum: %f\n",sum);
    //y_device[0] = 20;
}

__global__ void result_print(int rowA, int colA, MAT_VAL_TYPE *y_device){
    if(threadIdx.x == 0 &&threadIdx.y == 0 && blockIdx.x== 0&&blockIdx.y==0) {
        //printf("%d %d %d %d\n",threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
        for(int i=0;i<10;++i){
            for(int j=0;j<10;++j){
                printf("%2.0f ",y_device[i*colA+j]);
            }
            printf("\n");
        }
    }
}
int main(int argc, char ** argv){
	char  *filename;
    filename = argv[1];
    printf("MAT: -------------- %s --------------\n", filename);
    
 	struct timeval t1, t2;
    int rowA;
    int colA;
    int nnzA;
    int isSymmetricA;
    MAT_VAL_TYPE *csrValA, *csrValA_device;
    int *csrColIdxA, *csrColIdxA_device;
    MAT_PTR_TYPE *csrRowPtrA, *csrRowPtrA_device;
    
    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtrA, &csrColIdxA, &csrValA, filename);
    //make dense matrix with random numbers
	MAT_VAL_TYPE *x = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * colA*N);
    MAT_VAL_TYPE *x_device;
    //colA = 12;
	for (int i = 0; i < colA; i++){
        for(int j=0;j<N;++j){
		    x[i*N+j] = float((i*N+j)%10);
            x[i*N+j] = j;
        }
	}

    // compute reference results on a cpu core
    //rowA = 16;
	MAT_VAL_TYPE *y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA*N);
    //resizing for test
	for (int i = 0; i < rowA; i++){
        int initial_col = 0;
        int end_col = colA;
        for(int k=0;k<N;++k){
		    MAT_VAL_TYPE sum = 0;
		    for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++){
                if(csrColIdxA[j]>=colA) break;
		    	sum += csrValA[j] * x[csrColIdxA[j]*N+k];
		    }
		    y_golden[i*N+k] = sum;
        }
        /*
		for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++){
            for(;initial_col<MIN(csrColIdxA[j],end_col);++initial_col){
                printf("0 ");
            }
            if(csrColIdxA[j]>=end_col) break;
            printf("%.0f ",csrValA[j]);
            initial_col = csrColIdxA[j] +1;
		}
        for(;initial_col<end_col;++initial_col) printf("0 ");
        printf("\n");
        */
	}

	MAT_VAL_TYPE *y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA*N);
    MAT_VAL_TYPE *y_device;
    //memset(y, 0, sizeof(MAT_VAL_TYPE) * rowA*N);


    printf("  input matrix A: ( %i, %i ) nnz = %i\n", rowA, colA, nnzA);
    //cudaMemcpy(rowA_device, rowA, sizeof(int),cudaMemcpyHostToDevice);
    //cudaMemcpy(colA_device,colA, sizeof(int),cudaMemcpyHostToDevice);


    cudaMalloc((void**)&csrValA_device, nnzA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void**)&csrRowPtrA_device, (rowA + 1) * sizeof(MAT_PTR_TYPE));
    cudaMalloc((void**)&csrColIdxA_device, nnzA * sizeof(int));
    cudaMalloc((void**)&x_device,  colA*N * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void**)&y_device, rowA*N * sizeof(MAT_VAL_TYPE));
    //y[0]= 100;
    cudaMemcpy(csrValA_device, csrValA, nnzA * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(csrRowPtrA_device, csrRowPtrA, (rowA + 1) * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColIdxA_device, csrColIdxA, nnzA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(x_device, x, colA*N * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y, rowA*N * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    //y[1]= 100;
    // rowA = 8;
    // colA = 8;
    dim3 grid((colA+TJ-1)/TJ,(rowA+TI-1)/TI);//((rowA+TI-1)/TI,(N+TK-1)/TK); //(sp row block, dns col block)
    dim3 thread_block(TJ,TI); // row, col counts
    //y[0]=1200;
    //j_stream<<<grid,thread_block>>>(rowA, colA, csrRowPtrA_device, csrColIdxA_device, csrValA_device, x_device, y_device);
    
    my_tiling<<<grid,thread_block>>>(rowA, colA, csrRowPtrA_device, csrColIdxA_device, csrValA_device, x_device, y_device);
    printf(" grid y: %d, x: %d\n threadBlock y: %d, x: %d\n", grid.y,grid.x, thread_block.y, thread_block.x);
    result_print<<<grid,thread_block>>>(rowA, colA, y_device);
    //grid = (1,1);
    //thread_block = (rowA,N);
    //testing<<<grid,thread_block>>>(y_device);
    //copy_test<<<grid, thread_block>>>(y_device);
    cudaMemcpy(y, y_device,rowA*N * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
    int same_result = 1;
    int err_cnt = 0;
    for(int i=0;i<rowA&&same_result;++i){
        for(int j=0;j<N;++j){
            //printf("row: %d, col: %d, y: %f, y_golden: %f\n",i,j,y[i*N+j],y_golden[i*N+j]);
            if(y[i*N+j] != y_golden[i*N+j]){
                    err_cnt++;
                if (err_cnt>10){
                    same_result = 0;
                    break;
                }

                printf("*** error count: %d *** ",err_cnt);
                printf("row: %d, col: %d, y: %f, y_golden: %f\n",i,j,y[i*N+j],y_golden[i*N+j]);
            }
        }
    }
    if(same_result) printf("gpu and cpu computation results are same!!!\n");
    else printf("****gpu and cpu computation results are not same.....\n");
    cudaFree(csrValA_device);
    cudaFree(csrColIdxA_device);
    cudaFree(csrRowPtrA_device);
    cudaFree(x_device);
    cudaFree(y_device);
    return 0;
}