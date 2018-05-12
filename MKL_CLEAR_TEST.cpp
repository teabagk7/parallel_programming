#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <time.h>
#include <ctime>
#include <sys/time.h>
#include <fstream> 
#include <mkl.h>
#include <mkl_spblas.h>
#include <math.h>

#define shift 3
#define cache 512
//Filling in arrays A, LJ, LI, Rows_End
void CSR_Laplace(float * A, int * LJ, int * LI, int * ROWS_END , int size, int sqrt_size, int index_first_row, int index_last_row){
	int i = 0;
	*(LI++) = i;
	while (index_first_row < index_last_row + 1 && index_first_row < size){
		if (index_first_row >= sqrt_size){
			*(A++) = -1;
			*(LJ++) = index_first_row - sqrt_size    ;//
			i++;
		}
		if (index_first_row % sqrt_size != 0){
			*(A++) = -1;
			*(LJ++) = index_first_row - 1    ;//
			i++;
		}
		*(A++) = 4;
		*(LJ++) = index_first_row ;//
		i++;
		if (index_first_row % sqrt_size != sqrt_size - 1){
			*(A++) = -1;
			*(LJ++) = index_first_row + 1    ;//
			i++;
		}
		if (index_first_row < size - sqrt_size){
			*(A++) = -1;
			*(LJ++) = index_first_row + sqrt_size   ;//
			i++;
		}
		*(LI++) = i;
		*(ROWS_END++) = i;
		index_first_row++;
	}
	*(LI) = -1;
}
//vector initialization by 1
void single_vector(float * vector, int size){
	for (int i = 0; i < size; i++)
		vector[i] = 1;
}
//vector initialization by 0
void zero_vector(float * vector, int size){
	for (int i = 0; i < size; i++)
		vector[i] = 0;
}
//vector output in console
void print_vector(float * vector, int size){
	for (int i = 0; i < size; i++)
		std::cout << vector[i] << std::endl;
	std::cout << "\n" << "\n";
}
//Multiplication MATRIX(In CSR storage method) on vector(linear array)
void multiplication_matrix_on_vector(float * A, int * LJ, int * LI, float * vector, float * answer_vector, int size){
	int sum = 0;
	for (int i = 0; i <= size; i++){
		sum = 0;
		for (int j = LI[i]; j < LI[i + 1]; j++)
			sum += A[j] * vector[LJ[j]];
		answer_vector[i] = sum;
	}
}
double mysecond(){
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int main(int argc, char ** argv){
	std::ofstream fout;
	std::cout.precision(10);
	fout.open("clear_MKL_MAX(32)_THREAD_cache_512.txt");
	int threads = omp_get_max_threads();
	mkl_set_num_threads(threads);

	for (int i = 512; i < 5121; i += 512){
		int sqrt_size = i;
		int size = pow(i, 2);

		fout << "current size: " << i << " in power of 2 = " << size << "\n";

		int non_zero_amount = 5 * size - 4 * sqrt(size);
		float * A, * answer_vector, * vector;
		int * LJ, * LI, * ROWS_END;
		A = (float*)mkl_malloc(sizeof(float)* non_zero_amount, cache);
		vector = (float*)mkl_malloc(sizeof(float)* size, cache);
		answer_vector = (float*)mkl_malloc(sizeof(float)* size, cache);
		LJ = (int*)mkl_malloc(sizeof(int) * non_zero_amount, cache);
		LI = (int*)mkl_malloc(sizeof(int) * (size + 1), cache);
		ROWS_END = (int*)mkl_malloc(sizeof(int) * (size + 1), cache);

		//initialization
		zero_vector(answer_vector, size);
		single_vector(vector, size);
		CSR_Laplace(A, LJ, LI, ROWS_END, size, sqrt_size, 0, size);

		//data declaration
		sparse_matrix_t B;
		matrix_descr z;
		z.type = SPARSE_MATRIX_TYPE_GENERAL;
		sparse_status_t STATUS_CSR_B;
		
		//filling in B CSR matrix
		mkl_sparse_s_create_csr(&B, SPARSE_INDEX_BASE_ZERO, size, size, LI, ROWS_END, LJ, A);
		mkl_sparse_optimize(B);
		
		//
		mkl_free(LJ);
		mkl_free(LI);
		mkl_free(ROWS_END);
		/*-----------------------------parallel method---------------------------------------*/
		double		t;
		t = mysecond();

		//mkl_scsrgemv("N", &size, A, LI, LJ,vector,answer_vector);
		mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, B, z, vector, 0, answer_vector);

		t = (mysecond() - t);
		//timersub(&tval_after, &tval_before, &tval_result);
		fout << "parallel duration: " << t << "\n";

		mkl_sparse_destroy(B);
		
		LJ = (int*)mkl_malloc(sizeof(int)* non_zero_amount, cache);
		LI = (int*)mkl_malloc(sizeof(int)* (size + 1), cache);
		ROWS_END = (int*)mkl_malloc(sizeof(int)* (size + 1), cache);
		CSR_Laplace(A, LJ, LI, ROWS_END, size, sqrt_size, 0, size);
		zero_vector(answer_vector, size);
		single_vector(vector, size);
		/*-----------------------------consistent method---------------------------------------*/
		t = mysecond();

		multiplication_matrix_on_vector(A, LJ, LI, vector, answer_vector, size);
		
		t = (mysecond() - t);
		fout << "in series duration: " << t << "\n";
		
		mkl_free(A); 
		mkl_free(vector);
		mkl_free(answer_vector);
		mkl_free(LJ);
		mkl_free(LI);
		mkl_free(ROWS_END);
	}
	fout.close();
}
