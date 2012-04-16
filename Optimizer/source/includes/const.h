#define TRUE 1
#define FALSE 0

#define MAXSTRING 512

#define FLOODING 0
#define PROBABILISTIC_FLOODING 1
#define S1 2
#define S2 3
#define S3 4
#define INFOCOM 5
#define HYBRID 6
#define DISTANCE_PROBABILISTIC_FLOODING 7

#define RDUNIFORM 0
#define RD1 1
#define RD2 2
#define RD3 3
#define RD4 4
#define RD5 5
#define RDLOGISTIC 6

#define ALWAYSON -1
#define AVUNIFORM 0
#define AV1 1
#define AV2 2
#define AV3 3
#define AV4 4
#define AV5 5
#define AVLOGISTIC 6

//#define logistic(k,A,C,T,B,M) A + ( C ) / pow(( 1 + T * exp(-B * (id{k] - M) ) ), 1/T)
#define pblock(k) (1 - rho[k]) * pow(rho[k],buffer) / (1 - pow(rho[k],buffer+1)))
#define MIN(a,b) a<b?a:b
