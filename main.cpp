#include "mkl.h"
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <string.h>
#include <sstream>
#include "mpi.h"
#include <fstream>
//#include "lapacke_utils.h"

using namespace std;

void diagMatrix(int p, int k, float *m);
void ortBasis(int p, float *m);
void out(int p, float *m);
void choleskyFactorization(int p, float *m);
void copy(int p, float *from, float *to);
void samplesGenerate(int n, int p, float *mean, float *chol, float *res);
void wGenerate(int n, float* w);
void sampleCovarianceBootstrap(int n, int p, float* Samples, float* Result);
void sampleCovariance(int n, int p, float* Samples, float* Result);
void eigenDecomposition(int p, float* m, float *values);
void rProjector(int p, int r, float *vectors, float* projector);
float S(int n, int p, float *m1, float *m2);
float **alloc2d(int n, int m);

float **alloc2d(int n, int m)
{
    float *data = (float*) malloc(n*m*sizeof(float));
    float **array = (float **)malloc(n*sizeof(float *));
    for (int i=0; i<n; i++)
        array[i] = &(data[i*m]);
    return array;
}

struct aboutMatrix
{
    float* matrix;
    float* l;
    float* u;
    float* cholesky;
};

void diagMatrix(int p, int k, float *m)
{
    float numbers[5] = {36, 30, 25, 19, 15};
    int ind = 0;
    for (int i = 0; i < p; ++i)
    {
        for (int j = 0; j < p; ++j)
            m[i*p + j] = 0.0;
        if (k > 0)
        {
            m[i*p + i] = numbers[ind];
            ind++;
            k--;
        }
        else
            m[i*p + i] = 1.0 + 5.0*rand() / float(RAND_MAX);
    }
}

void ortBasis(int p, float *m)
{
for (int i = 0; i < p; i++)
        for(int j = 0; j < p; j++)
            m[i*p + j] = rand();
    float* tau = (float*) calloc(p, sizeof(float));
    LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, p, p, m, p, tau);
    LAPACKE_sorgqr(LAPACK_ROW_MAJOR, p, p, 1, m, p, tau);
    free(tau);
}

void out(int r, int c, float *m)
{
    //cout << "{" << endl;
    for (int i = 0; i < r; i++)
    {
        cout << "{ ";
        for (int j = 0; j < c; j++)
            cout << m[i*c + j] << " ";
        cout << "}" << endl;
    }
    //cout << "}" << endl;
}

//void choleskyFactorization(int p, float m[][p]);
void choleskyFactorization(int p, float *m)
{
    //A = U^T*U
    lapack_int q = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', p, m, p);
    //cout << q << endl;
}

void copy(int p, float *from, float *to)
{
    mkl_somatcopy('R', 'N', p, p, 1.0, from, p, to, p);
}

void samplesGenerate(int n, int p, float *mean, float *chol, float *res)
{
    VSLStreamStatePtr stream;
    int i = (int) rand();
    vslNewStream(&stream,VSL_BRNG_MCG31, i);
    vsRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER, stream, n, res, p, VSL_MATRIX_STORAGE_FULL, mean, chol);
}

void wGenerate(int n, float* w)
{
    float *mean = (float *) calloc(n, sizeof(float));
    float *cov = (float*) calloc(n * n, sizeof(float));
    for (int i = 1; i < n; i++)
    {
        mean[i] = 1;
        cov[i*n + i] = 1;
    }
    int i = (int) rand();
    VSLStreamStatePtr stream;
    vslNewStream(&stream,VSL_BRNG_MCG31, i);
    vsRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER, stream, 1, w, n, VSL_MATRIX_STORAGE_FULL, mean, cov);
    free(mean);
    free(cov);
}

void sampleCovarianceBootstrap(int n, int p, float* Samples, float* Result)
{
 //Result p*p
    float *W = (float *) calloc(n, sizeof(float));
    float *X = (float *) calloc(p, sizeof(float));
    wGenerate(n, W);
    for(int k = 0; k < n; k++)
    {
        for (int i = 0; i < p; i++)
        {
            X[i] = Samples[k*p + i];
        }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, p, p, 1, W[k], X, p, X, p, 1.0, Result, p);
    }
    free(W);
    free(X);
}

void sampleCovariance(int n, int p, float* Samples, float* Result)
{
 //Result p*p
    float *X = (float *) calloc(p, sizeof(float));
    for(int k = 0; k < n; k++)
    {
        for (int i = 0; i < p; i++)
        {
            X[i] = Samples[k*p + i];
        }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, p, p, 1, 1.0, X, p, X, p, 1.0, Result, p);
    }
    free(X);
}

void eigenDecomposition(int p, float* m, float *values) //decreasing orderб столбыц!
{
    LAPACKE_ssyevd(LAPACK_ROW_MAJOR, 'V', 'U', p, m, p, values);
}

void rProjector(int p, int r, float *vectors, float* projector)
{
    int num = p - r;
    float *X = (float *) calloc(p, sizeof(float));
    for (int i = 0; i < p; i++)
    {
        X[i] = vectors[p*i + num];
    }
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, p, p, 1, 1.0, X, p, X, p, 1.0, projector, p);
    free(X);   
}

float S(int n, int p, float *m1, float *m2)
{
    float *m = (float *) calloc(p * p, sizeof(float));
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++)
            m[i*p + j] = m1[i*p + j] - m2[i*p + j];
    float norm = LAPACKE_slange(LAPACK_ROW_MAJOR, 'F', p, p, m, p);
    free(m);
    return n*norm*norm;
}



int main(int argc, char **argv)
{

    int paral = 16;
    int p = 1000;
    int k = 4;
    int M = 3000;
    int r = 1; //s 1
    int N = 3; //48
    int nn = 14;
    int n_matr[14] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000};
    
    int rank, size, prev, next;
    int buf[2];
    MPI_Request reqs[4];
    MPI_Status stats[4];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    prev = rank - 1;
    next = rank + 1;
    if(rank==0) prev = size - 1;
    if(rank==size - 1) next = 0;

    srand(time(NULL));
    ofstream myfile;
    char intStr[2];
    if (rank < 10)
    {
        intStr[0] = '0';
        intStr[1] = '0' + rank;
    }
    else
    {
        intStr[0] = '1';
        intStr[1] = '0' + rank - 10;
    }
    char *txt = ".txt";
    char* str = strcat(intStr,txt);

    myfile.open(str);

    ofstream dim;
    ofstream cov;
    ofstream val;

    if (rank == 0){
        dim.open("dimensions.txt");
        cov.open("cov.txt");
        val.open("values.txt");
    }

    if (rank == 0)
    {
        dim << "p = " << p << endl;
        dim << "k = " << k << endl;
        dim << "M = " << M << endl;
        dim << "r = " << r << endl;
        dim << "N = " << N*paral << endl << endl;
        dim << "n = ";
        for (int i = 0; i < nn; i++)
        {
            dim << n_matr[i] << ", ";
        }
        dim << endl;
    }

    float *COV = (float *) calloc(p * p, sizeof(float));
    float *MEAN = (float *) calloc(p, sizeof(float));
    float *COV_Chol = (float *) calloc(p * p, sizeof(float));
    float *Basis = (float *) calloc(p * p, sizeof(float));
    float *COV_L = (float *) calloc(p * p, sizeof(float));

    //making covariance matrix
    ortBasis(p, Basis);

    diagMatrix(p, k, COV_L);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, p, p, p, 1.0, Basis, p, COV_L, p, 0.0, COV, p);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, p, p, p, 1.0, COV, p, Basis, p, 0.0, COV_L, p);
    copy(p, COV_L, COV);
    copy(p, COV, COV_Chol);
    choleskyFactorization(p, COV_Chol);
    free(Basis);
    free(COV_L);

    //дополнительные выводы
    if(rank == 0)
    {
        cov << "Cov matrix: " << endl;
        for (int i = 0; i < p; i++)
        {
            cov << "{ ";
            for (int j = 0; j < p; j++)
                cov << COV[i*p + j] << " ";
            cov << "}" << endl;
        }
        cov << endl;
    }

    float *values_pr = (float *) calloc(p, sizeof(float));
    float *vectors_pr = (float *) calloc(p * p, sizeof(float));

    copy(p, COV, vectors_pr);
    eigenDecomposition(p, vectors_pr, values_pr);

    if(rank ==0){
    myfile << endl << "Eigen values:" << endl;
    for (int i = 0; i < p; i++)
    {
        val << values_pr[i] << " ";
    }
    val << endl << endl;
    }
    
    free(values_pr);
    free(vectors_pr);



    // передаю n p, k, M, r, N, MEAN, COV, 

    //int ProcNum, ProcRank, tmp;////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //COV, COV_Chol

    //BOOTSTRAP
    srand(time(NULL) + rank);
    for (int num = 0; num < nn; num++)
    {
        int n = n_matr[num];
        float* NormsBootstrapWorld = (float *) calloc(N * M, sizeof(float));
        for (int i = 0; i < N; i++)
        {
            float *Samples = (float *) calloc(n * p, sizeof(float)); //stroki
            float *COV_Sample = (float *) calloc(p * p, sizeof(float));
            float *ProjectorS = (float *) calloc(p * p, sizeof(float));
            float *values = (float *) calloc(p, sizeof(float));
            float *vectors = (float *) calloc(p * p, sizeof(float));
            
            samplesGenerate(n, p, MEAN, COV_Chol, Samples);
            sampleCovariance(n, p, Samples, COV_Sample);
            copy(p, COV_Sample, vectors);
            eigenDecomposition(p, vectors, values);
            rProjector(p, r, vectors, ProjectorS);

            free(COV_Sample);
            free(values);
            free(vectors);

            for (int j = 0; j < M; j++)
            {
                float *COV_SampleBootstrap = (float *) calloc(p * p, sizeof(float));            
                float *values = (float *) calloc(p, sizeof(float));
                float *vectors = (float *) calloc(p * p, sizeof(float));
                float *ProjectorSB = (float *) calloc(p * p, sizeof(float));

                sampleCovarianceBootstrap(n, p, Samples, COV_SampleBootstrap);
                copy(p, COV_SampleBootstrap, vectors);
                eigenDecomposition(p, vectors, values); //столбцы!!!
                rProjector(p, r, vectors, ProjectorSB);

                float s = S(n, p, ProjectorS, ProjectorSB);
                NormsBootstrapWorld[i*M + j] = s;

                free(COV_SampleBootstrap);
                free(values);
                free(vectors);
                free(ProjectorSB);
            }
            free(Samples);
            free(ProjectorS);
        }

        myfile << "n = " << n << endl;
        myfile << "BOOTSTRAP NM" << endl;

        for (int i = 0; i < N; i++)
        {
            myfile << "{ ";
            for (int j = 0; j < M; j++)
                myfile << NormsBootstrapWorld[i*M + j] << " ";
            myfile << "}" << endl;
        }

        myfile << endl;            
        free(NormsBootstrapWorld);
    }
  //  out(M, N, NormsBootstrapWorld);

    float *values = (float *) calloc(p, sizeof(float));
    float *vectors = (float *) calloc(p * p, sizeof(float));
    float *ProjectorR = (float *) calloc(p * p, sizeof(float));

    copy(p, COV, vectors);
    eigenDecomposition(p, vectors, values);
    rProjector(p, r, vectors, ProjectorR);

    free(values);
    free(vectors);


    for (int num = 0; num < nn; num++)
    {
        int n = n_matr[num];
        float *NormsRealWorld = (float *) calloc(N * M, sizeof(float));
        for (int i = 0; i < 1; i++)
        {
            for (int j = 0; j < M; j++)
            {
                float *Samples = (float *) calloc(n * p, sizeof(float)); //stroki
                float *COV_Sample = (float *) calloc(p * p, sizeof(float));
                float *ProjectorS = (float *) calloc(p * p, sizeof(float));
                float *values = (float *) calloc(p, sizeof(float));
                float *vectors = (float *) calloc(p * p, sizeof(float));
                
                samplesGenerate(n, p, MEAN, COV_Chol, Samples);
                sampleCovariance(n, p, Samples, COV_Sample);
                copy(p, COV_Sample, vectors);
                eigenDecomposition(p, vectors, values);
                rProjector(p, r, vectors, ProjectorS);

                float s = S(n, p, ProjectorS, ProjectorR);
                NormsRealWorld[i*M + j] = s;

                free(COV_Sample);
                free(values);
                free(vectors);
                free(ProjectorS);
            }
        }
        myfile << "n = " << n << endl;
        myfile << "REAL 1M" << endl;
        for (int i = 0; i < 1; i++)
        {
            myfile << "{ ";
            for (int j = 0; j < M; j++)
                myfile << NormsRealWorld[i*M + j] << " ";
            myfile << "}" << endl;
        }
        myfile << endl;
        free(NormsRealWorld);

    }
    MPI_Barrier(MPI_COMM_WORLD);


    myfile << endl;
//    out(M, N, NormsRealWorld);

    free(ProjectorR);
    free(COV);
    free(MEAN);
    free(COV_Chol);
    MPI_Finalize();

}
