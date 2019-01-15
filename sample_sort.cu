#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <cuda.h>
#include<assert.h>

#define N 4194304
#define per_block 1024

__global__ void sample_sort(int *A)  //initial local sorting
{
	__shared__ int loc[per_block];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
 	int k = threadIdx.x;
 	loc[k] = A[i];
 	__syncthreads();
 	int j;
 	for(j = 0; j <  per_block/2; j++)
 	{
 		if( k%2==0 && k < per_block - 1)
 		{
 			if(loc[k] > loc[k + 1])
 			{
 				int keep = loc[k];
 				loc[k] = loc[k + 1];
 				loc[k + 1] = keep;
 			}
 		}
 		__syncthreads();

 		if( k%2 == 1 && k < per_block - 1)
 		{
 			if(loc[k] > loc[k+1])
 			{
 				int temp = loc[k];
 				loc[k] = loc[k + 1];
 				loc[k + 1] = temp;
 			}
		}
 		__syncthreads();
 	}
 	A[i] = loc[threadIdx.x];
}

__global__ void final_merge( int *A, int* S)  //final sorting
{

	int lower_limit = S[blockIdx.x];
	int upper_limit = S[blockIdx.x + 1]; //taking splitters
	int k = threadIdx.x;
	__shared__ int count[1025];
	count[0] = 0;
	int i, count_element = 0;

	for(i = 0; i < 4096; i++)
	{
		int temp = A[i +4096*k];
		if(temp >= lower_limit && temp < upper_limit)
			count_element++;
	}
	count[k+1] = count_element;
	__syncthreads();
	if(k == 0)
	{
		for(i = 2; i < 1025; i++)
			count[i] = count[i]+count[i-1];
	}
	__shared__ int my_block[5000];
	my_block[4*k] = N+2;
	my_block[4*k + 1] = N+2;
	my_block[4*k + 2] = N+2;
	my_block[4*k + 3] = N+2;
	__syncthreads();
	int g = 0;
	int index = count[k];
	for(i = 0; i < 4096; i++)
	{
		int keep =A[i+4096*k];
		if(keep>= lower_limit && keep < upper_limit)
		{
			my_block[index+g] = keep;
			g++;
		}
	}

	__shared__ int final[5000]; //taking an array size of 5000 in shared memory
	final[4*k] = N+2;
	final[4*k + 1] = N+2;
	final[4*k + 2] = N+2;
	final[4*k + 3] = N+2;
	__syncthreads();
	int fin_count1 = 0, fin_count2 = 0, fin_count3 = 0, fin_count4 = 0;
	int first = my_block[4*k], second = my_block[4*k +1], third = my_block[4*k +2], forth = my_block[4*k + 3];
	for(i = 0; i < 4096; i++)
	{
		int check = my_block[i];
		if(first > check)
			fin_count1++;
		if(second > check)
			fin_count2++;
		if(third > check)
			fin_count3++;
		if(forth > check)
			fin_count4++;
	}

	final[fin_count1] = first;

	final[fin_count2] = second;

	final[fin_count3] = third;
	
	final[fin_count4] = forth;
	

	__syncthreads();

	if(final[4*k] == N+2)
	{
		int d = 4*k - 1;
		while(final[d]== N+2)
		{
			d = d - 1;
		}
		final[4*k] = final[d];
	}
	if(final[4*k+1] == N + 2)
	{
		int d = 4*k;
		while(final[d]== N+2)
		{
			d = d - 1;
		}
		final[4*k+1] = final[d];
	}
	if(final[4*k + 2] == N+2)
	{
		int d = 4*k+1;
		while(final[d]== N+2)
		{
			d = d - 1;
		}
		final[4*k+2] = final[d];
	}
	if(final[4*k + 3] == N+2)
	{
		int d = 4*k+2;
		while(final[d]== N+2)
		{
			d = d - 1;
		}
		final[4*k+3] = final[d];
	}
	__syncthreads();
}






void merge(int *arr, int l, int m, int r) //merge sort
{
    	int i, j, k;
    	int n1 = m - l + 1;
    	int n2 =  r - m;
 	int L[n1], R[n2];
    	for (i = 0; i < n1; i++)
        		L[i] = arr[l + i];
    	for (j = 0; j < n2; j++)
        		R[j] = arr[m + 1+ j];
 	i = 0; 
    	j = 0; 
    	k = l; 
    	while (i < n1 && j < n2)
    	{
        		if (L[i] <= R[j])
        		{
            		arr[k] = L[i];
            		i++;
        		}
        		else
        		{
            		arr[k] = R[j];
            		j++;
        		}
        		k++;
    	}
 	while (i < n1)
    	{
        		arr[k] = L[i];
        		i++;
        		k++;
    	}
    	while (j < n2)
    	{
        		arr[k] = R[j];
        		j++;
        		k++;
    	}
}
 
void mergeSort(int *arr, int left, int right)
{
    if (left < right)
    {
        int middle = left+(right-left)/2;
        mergeSort(arr, left, middle);
        mergeSort(arr, middle+1, right);
        merge(arr, left, middle, right);
    }
}







int main()
{
	struct timeval start_serial,end_serial, start_cuda, end_cuda;
	int* h_A = (int*) malloc (N*sizeof(int));
	int* m_A = (int*) malloc (N*sizeof(int));
	int* h_S = (int*) malloc (8192*sizeof(int));
	int i, z = 0;
	srand(time(NULL));
	for(i=0; i < N; i++)
	{
		int random = rand()%N+1;
		h_A[i] = random;
		m_A[i] = random;
		if(i%512 == 0)
		{
			h_S[z] = random;
			z=z+1;
		}
	}
	gettimeofday(&start_serial,NULL);
	mergeSort(m_A, 0, 2097152);
	gettimeofday(&end_serial,NULL);
	size_t size = N*sizeof(int);
	int *d_A, *d_S;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_S, 2049*sizeof(int));
	dim3 threads_per_block(1024);
	dim3 num_of_blocks(N/1024);
	gettimeofday(&start_cuda,NULL);
	sample_sort<<<num_of_blocks, threads_per_block>>>(d_A); // initial local sort kernel

	mergeSort(h_S, 0, 8191);
	int *h_F = (int*) malloc (2049* sizeof(int));
	h_F[0] = 0;
	h_F[2048] = N+1;
	int c, m =1;
	for(c = 1; c < 8192; c++)
	{
		if(c%4 == 0)
		{
			h_F[m] = h_S[c];
			m = m+1; 
		}
	}
	dim3 new_threads_per_block(1024);
	dim3 new_num_blocks(2048);
	cudaMemcpy(d_S, h_F, 2049*sizeof(int), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	final_merge<<<new_num_blocks, new_threads_per_block>>>(d_A, d_S); //final local sort kernel sending the splitters too
	cudaDeviceSynchronize();
	gettimeofday(&end_cuda,NULL);
	cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_S);
	printf("\nTime taken for calculation is %ld microseconds \n",(end_serial.tv_sec - start_serial.tv_sec)*1000000 + (end_serial.tv_usec - start_serial.tv_usec));
	printf("\nTime taken for calculation is %ld microseconds \n",(end_cuda.tv_sec - start_cuda.tv_sec)*1000000 + (end_cuda.tv_usec - start_cuda.tv_usec)); //time calculations
	return 0;

}