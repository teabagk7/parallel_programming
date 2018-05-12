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

void CSR_Laplace(int * A, int * LJ, int * LI, int size, int sqrt_size, int index_first_row, int index_last_row){

	int i = 0;
	*(LI++) = i;
	while (index_first_row < index_last_row + 1 && index_first_row < size){
		if (index_first_row >= sqrt_size){		//diag 5(bot)
			*(A++) = -1;
			*(LJ++) = index_first_row - sqrt_size;//
			i++;
		}
		if (index_first_row % sqrt_size != 0){	//diag 4
			*(A++) = -1;
			*(LJ++) = index_first_row - 1;
			i++;
		}
		*(A++) = 4;								//diag 3(main)
		*(LJ++) = index_first_row;
		i++;
		if (index_first_row % sqrt_size != sqrt_size - 1){
			*(A++) = -1;						//diag 2
			*(LJ++) = index_first_row + 1;
			i++;
		}
		if (index_first_row < size - sqrt_size){//diag 1(top)
			*(A++) = -1;
			*(LJ++) = index_first_row + sqrt_size;
			i++;
		}
		*(LI++) = i;
		index_first_row++;
	}
	*(LI) = -1;
}
void RenumerateMatrix(int * Renum_A, int * A, int * LJ, int * LI, int size, int sqrt_size){
	for (int i = 0; i <= size; i++){
		for (int k = LI[i]; k < LI[i + 1]; k++){
			if (k == LI[i + 1] - 1){			//diag 0
				if (i < size - sqrt_size)
					Renum_A[i+sqrt_size] = A[k];
			}
			if (LJ[k] == i - 1){				//diag 1
				Renum_A[i + size] = A[k];
			}
			if (LJ[k] == i)						//diag 2
				Renum_A[i + 2 * size] = A[k];
			if (LJ[k] == i + 1){				//diag 3
				Renum_A[i + 3 * size] = A[k];
			}
			if (k == LI[i] && i >= sqrt_size){	//diag 4
				Renum_A[i - sqrt_size + 4 * size] = A[k];
			}
		}
	}
}
void alt_RenumerateMatrix(int * Renum_A, int * A, int * LJ, int * LI, int size, int sqrt_size){
}
void unit_vector(int * vector, int size){
	for (int i = 0; i < size; i++){
		vector[i] = 1;
	}
	return;

}
void zero_vector(int * vector, int size){
	for (int i = 0; i < size; i++){
		vector[i] = 0;
	}
	return;
}
void print_vector(int * vector, int size){
	for (int i = 0; i < size; i++){
		std::cout << vector[i] << "  ";
	}
	std::cout << "\n" << "\n";
}
void fill_vectorSHIFTED(int * vector_from, int * vector_to,int from, int to,int num_of_block){
	for (int i = from; i < to; i++)
		vector_to[i + 8*num_of_block] += vector_from[i];
}
void NEW_MOV_ALG(int * Renum_A,int * A, int * LJ, int * LI, int * vector, int * answer_vector, int size, int sqrt_size){
	int i, j, k, l;
#pragma omp parallel for schedule(static, 100) private(j) shared(answer_vector, vector, Renum_A)
	for (i = 0; i < size; i ++){
		int sum = 0;
		for (j = 0; j < 5; j++){
			sum += Renum_A[i + j*size] * vector[i];
		}
		answer_vector[i] = sum;
	}

}
void multiplication_matrix_on_vector(int * A, int * LJ, int * LI, int * vector, int * answer_vector, int size){
	int sum = 0;
	for (int i = 0; i <= size; i++){
		sum = 0;
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
	fout.open("OMP_NON_CSR_MAX(32)_THREADS_cache_512.txt");
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
		int * A, *answer_vector, *vector;
		int * LJ, *LI, *Renum_A;
		A = (int*)mkl_malloc(sizeof(float)* non_zero_amount, cache);
		vector = (int*)mkl_malloc(sizeof(float)* size, cache);
		answer_vector = (int*)mkl_malloc(sizeof(float)* size, cache);
		LJ = (int*)mkl_malloc(sizeof(int)* non_zero_amount, cache);
		LI = (int*)mkl_malloc(sizeof(int)* (size + 1), cache);
		Renum_A = (int*)mkl_malloc(sizeof(int)* (5 * size), cache);

		zero_vector(Renum_A, 5 * size);
		zero_vector(answer_vector, size);
		unit_vector(vector, size);
		CSR_Laplace(A, LJ, LI, size, sqrt_size, 0, size);
		RenumerateMatrix(Renum_A, A, LJ, LI, size, sqrt_size);

		//parallel
		t=mysecond();

		NEW_MOV_ALG(Renum_A, A, LJ, LI, vector, answer_vector, size, sqrt_size);

		t = (mysecond() - t);
		fout << " Parallel duration is : " <<t<< "\n";	
		//getchar();
		mkl_free(Renum_A);
		//in series
		//CSR_Laplace(A, LJ, LI, size, sqrt_size, 0, size);
		zero_vector(answer_vector, size);
		unit_vector(vector, size);
		t=mysecond();

		multiplication_matrix_on_vector(A, LJ, LI, vector, answer_vector, size);

		t = (mysecond() - t);
		fout << " In series duration is : "<< t << "\n";
		mkl_free(A);
		mkl_free(vector);
		mkl_free(answer_vector);
		mkl_free(LI);
		mkl_free(LJ);
		
	}
	fout.close();
}