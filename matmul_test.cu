#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dirent.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <string.h>
#include "common.h"
#include "mmio_highlevel.h"

// #define N 128
// #define K 48
// #define TJ 2
// #define TI 2
// #define TK 32

#define val_type float

int N = 1024;
int K = 48;
int TJ = 2;
int TI = 2;
int TK = 32;


#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        fprintf(stderr,cudaGetErrorString(err));
    }
}
__global__ void testing(float* out ,int row, int col, int N){
    //printf("***************%d %d#############\n",threadIdx.y,threadIdx.y);
    if(threadIdx.x==0&&threadIdx.y==0&&blockIdx.x==0&&blockIdx.y==0){
        for(int i=0;i<N;++i){
            for(int j=0;j<N;++j){
                printf("%2.0f ",out[i*col+j]);
            }
            printf("\n");
        }
    }
}
__global__ void csr_mat_mul(int rowA, int colA, int *csrRowPtrA, int *csrColIdxA, val_type *csrValA, val_type *dnsMat, val_type *out , int TI, int TJ, int N){
    int row = (blockIdx.y*TI+threadIdx.y);
    int col = (blockIdx.x*TJ+threadIdx.x);
    if(row>=rowA || col >=colA) return;
    for(int i = csrRowPtrA[row];i<csrRowPtrA[row+1];++i){
        int c = csrColIdxA[i];
        if(col==c){
            for(int n=0;n<N;++n){
                atomicAdd(&out[row*N+n],csrValA[i]*dnsMat[c*N+n]);
            }
            break;
        }
        if(c>col) break;
    }
}

__global__ void csr_mat_mul_shared1(int rowA, int colA, int *csrRowPtrA, int *csrColIdxA, val_type *csrValA, val_type *dnsMat, val_type *out, int ti, int tj,int N){    
    int row = (blockIdx.y*ti+threadIdx.y);
    int col = (blockIdx.x*tj+threadIdx.x);
    if(row>=rowA || col >=colA) return;

    extern __shared__ val_type s_temp[];
    val_type *dnsMatShared = s_temp;
    //__shared__ val_type outShared[TI][N];
    int rowD = col;
    for(int i=threadIdx.y; i<N; i+=ti){
        dnsMatShared[threadIdx.x*N+i] = dnsMat[rowD*N+i];
    }
    for(int i = csrRowPtrA[row];i<csrRowPtrA[row+1];++i){
        int c = csrColIdxA[i];
        if(col==c){
            for(int n=0;n<N;++n){
                atomicAdd(&out[row*N+n],csrValA[i]*dnsMatShared[threadIdx.x*N+n]);
            }
            break;
        }
        if(c>col) break;
    }
}
void csr_init(char * filename, int first,char *func_type,int func_type_int, int targeted_execution, int user_ti, int user_tj){
    
    //host 
    int *csrRowPtrA, *csrColIdxA;
    val_type *csrValA, *dnsMat, *out;
    //device
    int *d_csrRowPtrA, *d_csrColIdxA;
    val_type *d_csrValA, *d_dnsMat, *d_out;

    int rowA;
    int colA;
    int nnzA;
    int isSymmetricA;
    FILE *fp;
    fp = fopen("matmul_results_test.csv","a");
    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtrA, &csrColIdxA, &csrValA, filename);
    char * mm_type;
    if(isSymmetricA) mm_type = "symmetric";
    else mm_type = "not symmetric";

    //init dense matrix
    int rowD = colA;
    int colD = N;
    dnsMat = (val_type*) malloc(sizeof(val_type)*rowD*colD);
    out = (val_type*) malloc(sizeof(val_type)*rowA*colD);
    //printf("row: %d col: %d\n",rowA, colA);
    for(int r=0;r<rowD;++r){
        for(int c = 0;c<colD;++c){
            dnsMat[r*colD+c] = float((r*colD+c)%10);
        }
    }

    //make golden result
    val_type *y_golden = (val_type *)malloc(sizeof(val_type) * rowA*N);
	for (int i = 0; i < rowA; i++){
        int initial_col = 0;
        int end_col = colA;
        for(int k=0;k<N;++k){
		    MAT_VAL_TYPE sum = 0;
		    for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++){
                if(csrColIdxA[j]>=colA) break;
		    	sum += csrValA[j] * dnsMat[csrColIdxA[j]*N+k];
                //atomicAdd(&sum)
		    }
		    y_golden[i*N+k] = sum;
        }
    }
    cudaMalloc((void**)&d_csrRowPtrA,sizeof(int)*(rowA+1));
    cudaMalloc((void**)&d_csrColIdxA,sizeof(int)*nnzA);
    cudaMalloc((void**)&d_csrValA,sizeof(val_type)*nnzA);
    cudaMalloc((void**)&d_dnsMat,sizeof(val_type)*rowD*colD);
    cudaMalloc((void**)&d_out,sizeof(val_type)*rowA*colD);

    cudaMemcpy(d_csrRowPtrA,csrRowPtrA,sizeof(int)*(rowA+1),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxA,csrColIdxA,sizeof(int)*nnzA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValA,csrValA,sizeof(val_type)*nnzA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_dnsMat,dnsMat,sizeof(val_type)*rowD*colD,cudaMemcpyHostToDevice);
    cudaMemcpy(d_out,out,sizeof(val_type)*rowA*colD,cudaMemcpyHostToDevice);

    if(first == 1)
        fprintf(fp, "application name, row, col, nnz, type, kernel type, TI, TJ, TK, time\n");
    
    if(targeted_execution == 1){
        TI = user_ti;
        TJ = user_tj;
            
        dim3 gridDim((colA+TJ-1)/TJ,(rowA+TI-1)/TI);
        dim3 blockDim(TJ, TI);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        // printf("shared 1, TI: %d, TJ: %d\n",TI,TJ);
        if(func_type_int == 0)
            csr_mat_mul<<<gridDim,blockDim>>>(rowA, colA, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_dnsMat, d_out,TI, TJ,N);
        // grid paramter for j stream tiling
        if(func_type_int == 1)
            csr_mat_mul_shared1<<<gridDim,blockDim,TJ*N*sizeof(val_type)>>>(rowA, colA, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_dnsMat, d_out,TI, TJ,N);
        
        cudaEventRecord(stop);
        // printf("TJ: %d\n", tj);
        
        cudaMemcpy(out,d_out,sizeof(val_type)*rowA*colD,cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("gpu time: %.10f\n",milliseconds);
        //printf("**************************\n");
        int same_result = 1;
        int err_cnt = 0;
        for(int i=0;i<rowA&&same_result;++i){
            for(int j=0;j<N;++j){
                //printf("row: %d, col: %d, y: %f, y_golden: %f\n",i,j,y[i*N+j],y_golden[i*N+j]);
                if(out[i*N+j] != y_golden[i*N+j]){
                    err_cnt++;
                    if (err_cnt>10){
                        same_result = 0;
                        break;
                    }
                    // printf("*** error count: %d *** ",err_cnt);
                    // printf("row: %d, col: %d, y: %f, y_golden: %f\n",i,j,out[i*N+j],y_golden[i*N+j]);
                }
            }
        }
        fprintf(fp, "%s, %d, %d, %d, %s, %s, %d, %d, %d,%.7f\n",filename,rowA,colA,nnzA,mm_type,func_type, TI,TJ,0,milliseconds);
        if(same_result) {
            // printf("gpu and cpu computation results are same!!!\n");
            // printf("TI: %d, TJ: %d, TI*TJ = %d\n",TI,TJ,TI*TJ);
            // printf("GI: %d, GJ: %d, GI*GJ = %d\n",gridDim.y,gridDim.x, gridDim.y*gridDim.x);
            fprintf(fp, "%s, %d, %d, %d, %s, %s, %d, %d, %d,%.7f\n",filename,rowA,colA,nnzA,mm_type,func_type, TI,TJ,0,milliseconds);
        }
        else {   
            CHECK_LAST_CUDA_ERROR();
            printf("\nTI: %d, TJ: %d, TI*TJ = %d ",TI,TJ,TI*TJ);
            printf("GI: %d, GJ: %d, GI*GJ = %d\n",gridDim.y,gridDim.x, gridDim.y*gridDim.x);
            // printf("gpu time: %.10f\n",milliseconds);
            // printf("****gpu and cpu computation results are not same.....\n");
        }
        memset(out,0,sizeof(val_type)*rowA*colD);
        cudaMemcpy(d_out, out,sizeof(val_type)*rowA*colD, cudaMemcpyHostToDevice);
    }
    else{
        for (int ti = 1; ti<=1024; ti *= 2){
            for(int tj = 1;tj*ti<=1024; tj *=2){
                TI = ti;
                TJ = tj;

                dim3 gridDim((colA+TJ-1)/TJ,(rowA+TI-1)/TI);
                dim3 blockDim(TJ, TI);
                if(gridDim.x>=2147483647 || gridDim.y >= 65535 ) {
                    // printf("%d %d %d %d\n",threadIdx.x, threadIdx.y, gridDim.x, gridDim.y);
                    continue;
                }
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
                // printf("shared 1, TI: %d, TJ: %d\n",TI,TJ);
                if(func_type_int == 0)
                    csr_mat_mul<<<gridDim,blockDim>>>(rowA, colA, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_dnsMat, d_out,TI, TJ,N);
                // grid paramter for j stream tiling
                if(func_type_int == 1)
                    csr_mat_mul_shared1<<<gridDim,blockDim,TJ*N*sizeof(val_type)>>>(rowA, colA, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_dnsMat, d_out,TI, TJ,N);

                cudaEventRecord(stop);
                // printf("TJ: %d\n", tj);

                cudaMemcpy(out,d_out,sizeof(val_type)*rowA*colD,cudaMemcpyDeviceToHost);
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                // printf("gpu time: %.10f\n",milliseconds);
                //printf("**************************\n");
                int same_result = 1;
                int err_cnt = 0;
                for(int i=0;i<rowA&&same_result;++i){
                    for(int j=0;j<N;++j){
                        if(out[i*N+j] != y_golden[i*N+j]){
                            err_cnt++;
                            if (err_cnt>10){
                                same_result = 0;
                                break;
                            }
                        }
                    }
                }
                fprintf(fp, "%s, %d, %d, %d, %s, %s, %d, %d, %d,%.7f\n",filename,rowA,colA,nnzA,mm_type,func_type, TI,TJ,0,milliseconds);
                if(same_result) {
                    fprintf(fp, "%s, %d, %d, %d, %s, %s, %d, %d, %d,%.7f\n",filename,rowA,colA,nnzA,mm_type,func_type, TI,TJ,0,milliseconds);
                }
                else {   
                    CHECK_LAST_CUDA_ERROR();
                }
                memset(out,0,sizeof(val_type)*rowA*colD);
                cudaMemcpy(d_out, out,sizeof(val_type)*rowA*colD, cudaMemcpyHostToDevice);
            }
        }
    }


    fclose(fp);
    //*/
    
    /*
    // // grid parameter for i stream tiling
    dim3 gridDim((N+TK-1)/TK, (rowD+TJ-1)/TJ);
    dim3 blockDim(TK, TJ);
    cudaEventRecord(start);
    csr_mat_mul2<<<gridDim,blockDim>>>(rowD, rowA, colD, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_dnsMat, d_out, TJ, TK,N);
    // csr_mat_mul_shared2<<<gridDim,blockDim>>>(rowA, colA, d_csrRowPtrA, d_csrColIdxA, d_csrValA, d_dnsMat, d_out, TJ, TK,N);
    cudaEventRecord(stop);
    //testing<<<gridDim,blockDim,N>>>(d_out,rowA, colA);
    */
    //test_on_cpu(out,colA);

    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIdxA);
    cudaFree(d_csrValA);
    cudaFree(d_dnsMat);
    cudaFree(d_out);

    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    free(dnsMat);
    free(out);
    printf("csr version end\n");
}

int main(int argc, char ** argv){
	char *filename;
    char *func_type;
    int func_type_int;
    int targeted_exectuion;
    char dirname[100] = "";

    filename = argv[1];
    func_type = argv[2];
    func_type_int = atoi(argv[3]);
    targeted_exectuion = atoi(argv[4]);
    //dirname = "~/";
    int first=1;
    
    struct dirent *de;
    DIR *dr = opendir(filename);
    // execute only user defined TI, TJ
    if (targeted_exectuion){

        csr_init(filename,first, func_type, func_type_int,targeted_exectuion,atoi(argv[5]),atoi(argv[6]));
    }
    else{
        while((de = readdir(dr)) != NULL){
            strcat(dirname,filename);
            strcat(dirname,de->d_name);
            strcat(dirname,"/");
            strcat(dirname,de->d_name);
            strcat(dirname,".mtx");
            // printf("%s\n",dirname);
    
            if(access(dirname,F_OK) != -1){
                csr_init(dirname,first,func_type,func_type_int, 0,0,0);
                printf("*****%s %d******\n",dirname,first);
                first = 0;
            }
            strcpy(dirname,"");
        }
        closedir(dr);
    }

    // csr_init(filename,first);
    //matmul_init();
}