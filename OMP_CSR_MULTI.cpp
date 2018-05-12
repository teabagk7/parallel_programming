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
#define load_ints_in_cache 16;

void CSR_Laplace(float * A, int * LJ, int * LI, int size, int sqrt_size, int index_first_row, int index_last_row){
	
	int i = 0;
	*(LI++) = i;
	while (index_first_row < index_last_row + 1 && index_first_row < size){
		if (index_first_row >= sqrt_size){
			*(A++) = -1;
			*(LJ++) = index_first_row - sqrt_size;//
			i++;
		}
		if (index_first_row % sqrt_size != 0){
			*(A++) = -1;
			*(LJ++) = index_first_row - 1;//
			i++;
		}
		*(A++) = 4;
		*(LJ++) = index_first_row;//
		i++;
		if (index_first_row % sqrt_size != sqrt_size - 1){
			*(A++) = -1;
			*(LJ++) = index_first_row + 1;//
			i++;
		}
		if (index_first_row < size - sqrt_size){
			*(A++) = -1;
			*(LJ++) = index_first_row + sqrt_size;//
			i++;
		}
		*(LI++) = i;
		index_first_row++;
	}
	*(LI) = -1;
}
void unit_vector(float * vector, int size){
	for (int i = 0; i < size; i++){
		vector[i] = 1;
	}
	return;

}
void zero_vector(float * vector, int size){
	for (int i = 0; i < size; i++){
		vector[i] = 0;
	}
	return;
}
void print_vector(float * vector, int size){
	for (int i = 0; i < size; i++){
		std::cout << vector[i] << "  ";
	}
	std::cout << "\n" << "\n";
}
void CSR_multiplication_matrix_on_vector(float * A, int * LJ, int * LI, float * vector, float * answer_vector, int size, int sqrt_size, int num_threads){
	int i, j, sum;
#pragma omp parallel for schedule(static, 100) private(j)
	//#pragma omp parallel for private(j) shared(answer_vector, vector, A, LI, LJ)
	for (i = 0; i <= size; i++){
		sum = 0;
		for (j = LI[i]; j < LI[i + 1]; j++){
			sum += A[j] * vector[LJ[j]];
		}
		answer_vector[i] = sum;
	}
}
void multiplication_matrix_on_vector(float * A, int * LJ, int * LI, float * vector, float * answer_vector, int size){
	int sum = 0;
	for (int i = 0; i <= size; i++){
		for (int j = LI[i]; j < LI[i + 1]; j++){
			sum += A[j] * vector[LJ[j]];
		}
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
	fout.open("OMP_CSR_MAX(32)_THREADS_cache_512.txt");
	double t;
	int sqrt_size;
	int size;
	int threads = omp_get_max_threads();
	omp_set_num_threads(threads);
	fout << "THREADS : " << threads<<"\n";
	for (int i = 512; i < 5121; i += 512){
		sqrt_size = i;
		size = pow(i, 2);
		fout << "current size: " << i << " in power of  2 = " << size << "\n";
		int non_zero_amount = 5 * size - 4 * sqrt(size);

		float * A, *answer_vector, *vector;
		int * LJ, *LI, *ROWS_END;
		A = (float*)mkl_malloc(sizeof(float)* non_zero_amount, cache);
		vector = (float*)mkl_malloc(sizeof(float)* size, cache);
		answer_vector = (float*)mkl_malloc(sizeof(float)* size, cache);
		LJ = (int*)mkl_malloc(sizeof(int)* non_zero_amount, cache);
		LI = (int*)mkl_malloc(sizeof(int)* (size + 1), cache);
		ROWS_END = (int*)mkl_malloc(sizeof(int)* (size + 1), cache);
		zero_vector(answer_vector, size);
		unit_vector(vector, size);
		CSR_Laplace(A, LJ, LI, size, sqrt_size, 0, size);
		//parallel
		t=mysecond();

		CSR_multiplication_matrix_on_vector(A, LJ, LI, vector, answer_vector, size, sqrt_size, threads);

		t = (mysecond() - t);
		fout << " Parallel duration is : " << t << "\n";
		
		//in series
		zero_vector(answer_vector, size);
		unit_vector(vector, size);
		t = mysecond();

		multiplication_matrix_on_vector(A, LJ, LI, vector, answer_vector, size);

		t = (mysecond() - t);
		fout << " In series duration is : " << t << "\n";

		mkl_free(A);
		mkl_free(vector);
		mkl_free(answer_vector);
		mkl_free(LI);
		mkl_free(LJ);
		mkl_free(ROWS_END);
	}
	fout.close();
}