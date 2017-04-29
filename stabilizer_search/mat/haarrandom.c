#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <haarrandom.h>
#include <time.h>


long double r(void){
        return (long double)rand()/(long double) RAND_MAX;
}

long double ** random_so2(void){
    srand(time(NULL));
    long double ** mat = malloc(2*sizeof(long double complex *));
    mat[0] = calloc(2, sizeof(long double complex));
    mat[1] = calloc(2, sizeof(long double complex));
    long double alpha = r()*2*M_PI;
    mat[0][0]=cosl(alpha);
    mat[1][1]=cosl(alpha);
    mat[0][1]=-1*sinl(alpha);
    mat[1][0]=sinl(alpha);
    return mat;
}

long double complex ** random_su2(void){
    srand(time(NULL));
    long double psi, chi, phi;
    phi = asinl(sqrtl(r()));
    chi = r()*2*M_PI;
    psi = r()*2*M_PI;
    long double complex **mat;
    mat = malloc(2*sizeof(long double complex));
    mat[0] = calloc(2, sizeof(long double complex));
    mat[1] = calloc(2, sizeof(long double complex));
    mat[0][0] = cexpl(psi*I)*cosl(phi);
    mat[0][1] = cexpl(chi*I)*sinl(phi);
    mat[1][0] = -1*cexpl(-1*chi*I)*sinl(phi);
    mat[1][1] = cexpl(-1*psi*I)*cosl(phi);
    return mat;
}
