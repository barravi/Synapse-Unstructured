/**
 SynOptimizer - A Synapse statistical model configuration optimizer and predictor
 
 Copyright (C) 2012 - Rossano Gaeta (gaeta@di.unito.it), Riccardo Loti (loti@di.unito.it)
 
 This file is part of SynOptimizer.
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with Nome-Programma.  If not, see <http://www.gnu.org/licenses/>.
 **/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../INCLUDES/const.h"

//#define DEBUG 0
//#define DEBUG_CALLS 0
extern FILE *efopen();
extern char *ecalloc();
extern void exit();
extern void free();
extern double pow();


#define acc 1e-6

int n;
int routing_strat;
int allconf; // If true iterates through all the permutations of sk, if false read which to use from synapse_file

double alpha;
double dx;

int *idpk  = NULL;
double *pk = NULL;

int *idsk  = NULL;
double *sk = NULL;

int *idpf  = NULL;
double *pf = NULL;

double *phit = NULL;
double *msg = NULL;

double (*HIT[7])(double);
double (*MSG[7])(double);
double (*QSR[7])(double);
double (*NSRC[7])(double,int);
// MSG[2](z) the recall
// Q1 .. Q6 must be declared and wrote with double parameter and return value
extern double G0(); extern double G1();
extern double F();
extern double M();
extern double N();
extern double Q();
extern double R();
extern double T();
extern double H();

extern double Q1(); extern double Q2(); extern double Q3();
extern double Q4(); extern double Q5(); extern double Q6();
extern double H1(); extern double H2(); extern double H3();
extern double H4(); extern double H5(); extern double H6();

#ifdef POIVEDIAMO
extern double HIT1(); extern double HIT2(); extern double HIT3();
extern double HIT4(); extern double HIT5(); extern double HIT6();
extern double QS1(); extern double QS2(); extern double QS3();
extern double QS4(); extern double QS5(); extern double QS6();
extern double NSRC1(); extern double NSRC2(); extern double NSRC3();
extern double NSRC4(); extern double NSRC5(); extern double NSRC6();
#endif

char degree_file[MAXSTRING];
char synapse_file[MAXSTRING];
char search_file[MAXSTRING];
char results_file[MAXSTRING];

int MAX_DEGREE;
int MAX_SYNAPSE;
int MAX_TTL;
int MAX_MESSAGES;

FILE *res_fp=NULL;

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double diff(double (f)(double), double z)
{
  return((f(z+dx) - f(z-dx)) / (2 * dx)); 
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
int max(int a, int b)
{
  return(a>b?a:b);
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
int min(int a, int b)
{
  return(a<b?a:b);
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
void disp(int *a, int size)
{
  int i=0;
  double avg=0.0;

  fprintf(res_fp, "%1.2f\t", (double)a[1]/n);    // Printing s1 element
 
  sk[1] = (double)a[1]/n;
  avg += sk[1] * idsk[1];
  for (i = 2; i <= size; i++)
  {
    fprintf(res_fp, "%1.2f\t", (double)a[i]/n);  // Printing sn elements
    sk[i] = (double)a[i]/n;
    avg += sk[i] * idsk[i];
  }
  sk[0] = avg;
  fprintf(res_fp, "\t\t%1.2f\t%1.4f\t%1.2f\n", diff(T, 1.0), 1-H(0), diff(F, 1.0));  // Printing T, phit and F
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
void N2N(int n,int i,int a,int b,int row,int col,int level,int cumsum,int *cell)
{ /* Init N2N */
  int j,tmp,ntmp,q,q2,jmin,jmax;
  int toploop,npart,p,p2;
  double res;
  
  //fprintf(stdout,"Init N2N\n");
  if (col != 0)
  { /* col != 0 */
    if (col == 1)
    { /* col == 1 */
      jmax = max(a,n-cumsum-b);
      jmin = min(b,n-cumsum-a);       
      if (allconf)                    // Cycle all configurations...
      {
        for (j = jmax; j <= jmin; j++)
        {
          cell[i-1] = j; cell[i] = n-cumsum-j;
          //disp(cell, n); 
          disp(cell, MAX_SYNAPSE);
        }
      }
      else                            // ... or just a specified one.
      {
        for (j = 1; j <= MAX_SYNAPSE; j++)
        {
          fprintf(res_fp, "%1.2f\t", sk[j]);
        }
        fprintf(res_fp, "\t\t%1.2f\t%1.4f\t%1.2f\n", diff(T, 1.0), 1-H(0), diff(F, 1.0));
      }
    } /* col == 1 */
    else
    { /* col != 1 */
      cell[level+1] = a;
      tmp = cumsum + a;
      res = (n - tmp) / (double)(i-level-1);
      ntmp = round((n - tmp) / (double)(i-level-1));
      //fprintf(stdout, "RES IS %g and ROUND is %d\n", res, ntmp);
      if (a <= ntmp && ntmp <= b && cell[level+1])
      {
       N2N(n, i, a, b, row-a+1, col-1, level+1, tmp, cell);
      }
      else
      {
        for (q = 1; q <= min((b-a),(row-a)-(col-1)); q++)
        {
          cell[level+1] = cell[level+1] + 1;
          tmp = tmp + 1;
          res = (n - tmp) / (double)(i-level-1);
          ntmp = round((n - tmp) / (double)(i-level-1));
          //fprintf(stdout,"RES IS %g and ROUND is %d\n",res,ntmp);
          if (a <= ntmp && ntmp <= b && cell[level+1])
          {
            q2 = q;
            q = min((b-a), (row-a)-(col-1));
            N2N(n, i, a, b, row-a-q2+1, col-1, level+1, tmp, cell);
          }
        }
      }
    } /* col != 1 */
  } /* col != 0 */
  else
  { /* col == 0 */
    fprintf(stdout,"Col is equal 0, granularity is %d\n",n);
    cell[1] = n;
    disp(cell, MAX_SYNAPSE);
  } /* col == 0 */

  if (level>0 && row>1)
  {
    cell[level] = cell[level]+1;
    cumsum = cumsum + 1;
    if(cell[level]<a)
    {
      cumsum = cumsum - cell[level] + a;
      cell[level] = a;
      row = row + cell[level] - a;
    }
    toploop = min(b-cell[level],row-col-1);
    res = (n-cumsum)/(double)(i-level);
    npart = round((n-cumsum)/(double)(i-level));
    //fprintf(stdout,"RES IS %g and ROUND is %d\n",res,npart);
    if (a<=npart && npart<=b && cell[level]<=b)
    {
      N2N(n, i, a, b, row-1, col, level, cumsum, cell);
    }
    else
    {
      for (p=1; p<=toploop; p++)
      {
        cell[level] = cell[level] + 1;
        cumsum = cumsum + 1;
        res = (n-cumsum) / (double)(i-level);
        npart = round((n-cumsum) / (double)(i-level));
        //fprintf(stdout,"RES IS %g and ROUND is %d\n",res,npart);
        if (a<=npart && npart<=b && cell[level]<=b)
        {
          p2 = p;
          p = toploop;
          N2N(n, i, a, b, row-p2, col, level, cumsum, cell);
        }
      }
    }
  }
  //fprintf(stdout,"End N2N\n");
} /* End N2N */

void jdoric(int n, int mink, int maxk, int a, int b)
{/* Init jdoric */
/*
%RIC :
%   Generates restricted and unrestricted integer compositions 
%   n = integer to be partitioned 
%   kmin = min. no. of summands 
%   kmax = max. no. of summands
%   a = min. value of summands
%   b = max.value of summands
%
% from : "A Unified Approach to Algorithms Generating Unrestricted
%           and Restricted Integer Compositions and Integer Partitions"
%   J. D. OPDYKE, J. Math. Modelling and Algorithms (2009) V9 N1, p.53 - 97
% 
% Matlab implementation :
% Theophanes E. Raptis, DAT-NCSRD 2010
% http://cag.dat.demokritos.gr
% rtheo@dat.demokritos.gr
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/
  int *cell = (int*)ecalloc(n+1, sizeof(int));
  int rowdec,i,in;

  rowdec = 0;
  for (i = mink; i <= maxk; i++)
  {
    in = n / i;
    if (a > 1)
    {
      rowdec = i;
    }
    else if ((a <= in) && (in <= b))
    {
      N2N(n, i, a, b, n-1-rowdec, i-1, 0, 0, cell);
    }
  }
  free(cell);
}/* End jdoric */

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
void read_distribution(char *file_name, int max, int *items, double *prob)
{
  FILE *fp = NULL;
  char Buf[MAXSTRING];
  int k, val;
  double p, avg = 0;

  #ifdef DEBUG_CALLS
  fprintf(stdout, "Init read_distribution\n");
  #endif
  fp = efopen(file_name, "r");
  fgets(Buf, MAXSTRING , fp);
  #ifdef DEBUG
  //fprintf(stdout, "Read %s\n", Buf);
  #endif
  for (k = 1; k <= max; k++)
  {
    fgets(Buf, MAXSTRING , fp);
    #ifdef DEBUG
    //fprintf(stdout, "Read %s", Buf);
    #endif
    sscanf(Buf, "%d %lg\n", &val, &p);
    items[k] = val;
    prob[k] = p;
    avg += p * val;
    #ifdef DEBUG
    //fprintf(stdout, "Numbers are %d %lg\n", val, prob);
    #endif
  }
  prob[0] = avg;
  fclose(fp);  
  #ifdef DEBUG_CALLS
  fprintf(stdout, "End read_distribution\n");
  #endif
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
int compute_max(char *file_name)
{
  FILE *fp = NULL;
  char Buf[MAXSTRING];
  int max;

  #ifdef DEBUG_CALLS
  fprintf(stdout, "Init compute_max\n");
  #endif
  fp = efopen(file_name, "r");
  fgets(Buf, MAXSTRING , fp);
  #ifdef DEBUG
  fprintf(stdout, "Read %s\n", Buf);
  #endif
  sscanf(Buf, "# %d\n", &max);
  #ifdef DEBUG
  //fprintf(stdout, "Numbers is %d\n", max);
  #endif
  fclose(fp);  
  #ifdef DEBUG_CALLS
  fprintf(stdout, "End compute_max\n");
  #endif
  return(max);
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
void parse_options(int argc, char **argv)
{
  int optr = 1;
  // Begin default values for parameters
  MAX_SYNAPSE = 10;   // 10 overlays
  MAX_TTL = 3;        // TTL 3
  alpha = 0.0001;     // Resource distribution 10^-4
  dx = 1e-9;          // 10^-9 dx for numerical integration
  n = 20;             // 20 (0.05 steps) granularity for allconf search
  routing_strat = 0;  // 1/k default routing
  allconf = 1;        // Search all configurations
  // End default values for parameters
  #ifdef DEBUG_CALLS
  fprintf(stdout, "Init parse_options\n");
  #endif
  while (optr < argc)
  {
    if ((strcmp(argv[optr], "-d") == 0) || (strcmp(argv[optr], "--degree") == 0))
    {
      optr++;
      sprintf(degree_file, "%s", argv[optr]);
      fprintf(stdout, "Degree file is %s.\n", degree_file);
    }
    else if ((strcmp(argv[optr], "-o") == 0) || (strcmp(argv[optr], "--overlays") == 0))
    {
      optr++;
      MAX_SYNAPSE = atoi(argv[optr]);
      fprintf(stdout, "Number of overlays is %i.\n", MAX_SYNAPSE);
    }
    else if ((strcmp(argv[optr], "-w") == 0) || (strcmp(argv[optr], "--write_file") == 0))
    {
      optr++;
      sprintf(results_file, "%s", argv[optr]);
      fprintf(stdout, "Result file is %s.\n", results_file);
    }
    else if ((strcmp(argv[optr], "-t") == 0) || (strcmp(argv[optr], "--ttl") == 0))
    {
      optr++;
      MAX_TTL=atoi(argv[optr]);
    }
    else if ((strcmp(argv[optr], "-a") == 0) || (strcmp(argv[optr], "--alpha") == 0))
    {
      optr++;
      alpha = atof(argv[optr]);
    }
    else if ((strcmp(argv[optr], "-x") == 0) || (strcmp(argv[optr], "--dx") == 0))
    {
      optr++;
      dx = atof(argv[optr]);
    }
    else if ((strcmp(argv[optr], "-g") == 0) || (strcmp(argv[optr], "--granularity") == 0))
    {
      optr++;
      n = atoi(argv[optr]);
    }
    else if ((strcmp(argv[optr], "-r") == 0) || (strcmp(argv[optr], "--routing") == 0))
    {
      optr++;
      routing_strat = atoi(argv[optr]);
    }
    else if ((strcmp(argv[optr], "-v") == 0) || (strcmp(argv[optr], "--verbose") == 0))
    {
      //TODO: To implement verbose flag in source code
    }
    else if ((strcmp(argv[optr], "-c") == 0) || (strcmp(argv[optr], "--conf") == 0))
    {
      allconf = 0;
      optr++;
      sprintf(synapse_file, "%s", argv[optr]);
    }
    else
    {
      fprintf(stdout, " - Synapse Optimizer - \n\n");
      fprintf(stdout, "Error: parsing parameters!\nParameter %s is not valid.\n\n", argv[optr]);
      //TODO: add usage
      exit(1);
    }
    optr++;
  }
  MAX_MESSAGES = MAX_SYNAPSE + 1; // TODO: For now defined here like so
  #ifdef DEBUG_CALLS
  fprintf(stdout, "End parse_options\n");
  #endif
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
int main(int argc, char **argv)
{
  char filename[MAXSTRING];
  FILE *ofp = NULL;
  int k;
  double average_overload;

  fprintf(stdout, "SynOptimizer - Synapse Optimizer (ver. 20120416-1613)\n");
  fprintf(stdout, "Copyright (C) 2012 - R. Gaeta, R. Loti\n\n");
  fprintf(stdout, "This program comes with ABSOLUTELY NOWARRANTY.\n");
  fprintf(stdout, "This is free software, and you are welcome to redistribute it\n");
  fprintf(stdout, "under certain conditions.\n\n");

  if (argc < 2)
  {
    fprintf(stdout, "Error: not enough parameters!\n\n");
    fprintf(stdout, "Usage: %s -d <degree_file> -w <result_file> [<parameter> <parameter_value>]\n", argv[0]);
    fprintf(stdout, "\t-d or --degree:      the degree distribution file\n");
    fprintf(stdout, "\t-w or --write_file:  the result output file\n");
    fprintf(stdout, "\t-o or --overlays:    the number of distinct overlays [Default: 10]\n");
    fprintf(stdout, "\t-t or --ttl:         the TTL of each query sent [Default: 3]\n");
    fprintf(stdout, "\t-a or --alpha:       the density of the requested resource [Default: 0.0001]\n");
    fprintf(stdout, "\t-x or --dx:          the dx for the numerical derivate calculation [Default: 1e-9]\n");
    fprintf(stdout, "\t-g or --granularity: the granularity of the searched permutations [Default: 20]\n");
    fprintf(stdout, "\t-r or --routing:     the node routing strategy in sending messages [Default: 0]\n");
    fprintf(stdout, "\t-v or --verbose:     set the console output to verbose [Default: false]\n");
    fprintf(stdout, "\t-c or --conf:        specify a file containing the configuration to be used [Default: none - all conf]\n\n");
    exit(1);
  }
  parse_options(argc, argv);


  MAX_DEGREE = compute_max(degree_file);
  /*if (!allconf)
  {
    MAX_SYNAPSE = compute_max(synapse_file); // Already defined by ovls number
  }*/

  idpk = (int *)ecalloc(MAX_DEGREE + 1, sizeof(int));
  pk = (double *)ecalloc(MAX_DEGREE + 1, sizeof(double));
  idsk = (int *)ecalloc(MAX_SYNAPSE + 1, sizeof(int));
  sk = (double *)ecalloc(MAX_SYNAPSE + 1, sizeof(double));
  idpf = (int *)ecalloc(MAX_SYNAPSE + 1, sizeof(int));
  pf = (double *)ecalloc(MAX_SYNAPSE + 1, sizeof(double));
  phit = (double *)ecalloc(MAX_TTL + 1, sizeof(double));
  msg = (double *)ecalloc(MAX_TTL + 1, sizeof(double));

  read_distribution(degree_file, MAX_DEGREE, idpk, pk);
  if (!allconf)
  {
    read_distribution(synapse_file, MAX_SYNAPSE, idsk, sk);
  }
  else
  {
    for (k=1; k<=MAX_SYNAPSE; k++)
    {
      idsk[k] = k;
    }
  }
  //read_distribution(search_file, MAX_SYNAPSE, idpf, pf);

  for (k=1; k<=MAX_SYNAPSE; k++)
  {
    switch (routing_strat)
    {
      case 0: // 1/k routing strategy
        pf[k] = 1.0 / idsk[k];
        break;
      case 1: // MAX_MESSAGES strategy //TODO Check it
        pf[k] = (MAX_MESSAGES - idsk[k]) / (MAX_MESSAGES - 1);
        break;
      case 2: // zmax strategy
        //NOTES: zmax is max messages wanted, z neighbors of a node
        if (((2*pk[0]) / (pk[0] * idsk[k])) < 1.0)
        {
          pf[k] = (2*pk[0]) / (pk[0] * idsk[k]);
        }
        else
        {
          pf[k] = 1.0;
        }
        //pf[k] = min(1, zmax / (pk[0] * idsk[k]));
        break;
      case 3: // Flooding strategy
        pf[k] = 1.0;
        break;
      case 4: // Fixed flooding strategy
        pf[k] = 0.75;
        break;
      default:
        pf[k] = 1.0 / idsk[k];
        break;
    }
  }
  MSG[0] = NULL; MSG[1] = Q1; MSG[2] = Q2; MSG[3] = Q3;
  MSG[4] = Q4; MSG[5] = Q5; MSG[6] = Q6;
  HIT[0] = NULL; HIT[1] = H1; HIT[2] = H2; HIT[3] = H3;
  HIT[4] = H4; HIT[5] = H5; HIT[6] = H6;
  res_fp = efopen(results_file, "w");
  fprintf(res_fp, "s1");
  for (k=2; k <= MAX_SYNAPSE; k++)
  {
    fprintf(res_fp, "\ts%d", k);
  }
  fprintf(res_fp, "\t\t\tT\tphit\tF\n");
  jdoric(n, MAX_SYNAPSE, MAX_SYNAPSE, 1, n);
  fclose(res_fp);

  /*fprintf(stdout,"Is G0 a pdf? %lg\n",G0(1.0));
  fprintf(stdout,"Is G1 a pdf? %lg\n",G1(1.0));
  fprintf(stdout,"Is F a pdf? %lg\n",F(1.0));
  fprintf(stdout,"Is M a pdf? %lg\n",M(1.0));
  fprintf(stdout,"Is N a pdf? %lg\n",N(1.0));
  fprintf(stdout,"Is Q a pdf? %lg\n",Q(1.0));
  fprintf(stdout,"Is R a pdf? %lg\n",R(1.0));
  fprintf(stdout,"Is T a pdf? %lg\n",T(1.0));
  fprintf(stdout,"Is H a pdf? %lg\n",H(1.0));
  fprintf(stdout,"Is Q1 a pdf? %lg\n",Q1(1.0));
  fprintf(stdout,"Is Q2 a pdf? %lg\n",Q2(1.0));
  fprintf(stdout,"Is Q3 a pdf? %lg\n",Q3(1.0));
  fprintf(stdout,"Is Q4 a pdf? %lg\n",Q4(1.0));
  fprintf(stdout,"Is Q5 a pdf? %lg\n",Q5(1.0));
  fprintf(stdout,"Is Q6 a pdf? %lg\n",Q6(1.0));
  fprintf(stdout,"Is H1 a pdf? %lg\n",H1(1.0));
  fprintf(stdout,"Is H2 a pdf? %lg\n",H2(1.0));
  fprintf(stdout,"Is H3 a pdf? %lg\n",H3(1.0));
  fprintf(stdout,"Is H4 a pdf? %lg\n",H4(1.0));
  fprintf(stdout,"Is H5 a pdf? %lg\n",H5(1.0));
  fprintf(stdout,"Is H6 a pdf? %lg\n",H6(1.0));
  fprintf(stdout, "Avg G0 is %lg\n", diff(G0,1.0));
  fprintf(stdout, "Avg G1 is %lg\n", diff(G1,1.0));
  fprintf(stdout, "Avg F is %lg\n", diff(F,1.0));
  fprintf(stdout, "Avg M is %lg\n", diff(M,1.0));
  fprintf(stdout, "Avg N is %lg\n", diff(N,1.0));
  fprintf(stdout, "Avg Q is %lg\n", diff(Q,1.0));
  fprintf(stdout, "Avg R is %lg\n", diff(R,1.0));
  fprintf(stdout, "Avg T is %lg\n", diff(T,1.0));
  fprintf(stdout, "Avg H is %lg\n", diff(H,1.0));
  fprintf(stdout, "Avg Q1 is %lg\n", diff(Q1,1.0));
  fprintf(stdout, "Avg Q2 is %lg\n", diff(Q2,1.0));
  fprintf(stdout, "Avg Q3 is %lg\n", diff(Q3,1.0));
  fprintf(stdout, "Avg Q4 is %lg\n", diff(Q4,1.0));
  fprintf(stdout, "Avg Q5 is %lg\n", diff(Q5,1.0));
  fprintf(stdout, "Avg Q6 is %lg\n", diff(Q6,1.0));
  fprintf(stdout, "Avg H1 is %lg\n", diff(H1,1.0));
  fprintf(stdout, "Avg H2 is %lg\n", diff(H2,1.0));
  fprintf(stdout, "Avg H3 is %lg\n", diff(H3,1.0));
  fprintf(stdout, "Avg H4 is %lg\n", diff(H4,1.0));
  fprintf(stdout, "Avg H5 is %lg\n", diff(H5,1.0));
  fprintf(stdout, "Avg H6 is %lg\n", diff(H6,1.0));
  fprintf(stdout, "It seems that phit=%lg\n", 1-H(0));*/
  exit(1);

  #ifdef DAFINIRE 
  HIT[0] = NULL; HIT[1] = HIT1; HIT[2] = HIT2; HIT[3] = HIT3;
  HIT[4] = HIT4; HIT[5] = HIT5; HIT[6] = HIT6;
  QSR[0] = NULL; QSR[1] = QS1; QSR[2] = QS2; QSR[3] = QS3;
  QSR[4] = QS4; QSR[5] = QS5; QSR[6] = QS6;
  NSRC[0] = NULL; NSRC[1] = NSRC1; NSRC[2] = NSRC2; NSRC[3] = NSRC3;
  NSRC[4] = NSRC4; NSRC[5] = NSRC5; NSRC[6] = NSRC6;
  for (MAX_TTL = 1; MAX_TTL <= 4; MAX_TTL++)
  { /* For each TTL */
    sprintf(filename, "%s-ttl-%d", res_name, MAX_TTL);
    ofp = efopen(filename, "w");
    for(lambda_gen = 0.00005; lambda_gen < 0.00010; lambda_gen+=0.00001)
    { /* For each lambda */
      fprintf(ofp, "####### CASE %lg\n", lambda_gen);
      init_availability();
      fill_query_algorithm();
      fill_resource_placement();
      iter = 1;
      do
      {
        compute_average_gamma_();
        compute_average_pr_x_a();
        compute_performance_indexes();
        compute_new_ak();
        average_overload = 0.0;
        for (k = 1; k <= MAX_DEGREE; k++)
        {
          average_overload += pk[k] * (1 - a[k]);
        }
        fprintf(ofp, "%1.6f %lg %lg %lg %d %lg\n", lambda_gen, msg[MAX_TTL], phit[MAX_TTL], average_overload, iter, maxerr);
        iter++;
      }
      while (maxerr > acc && iter < MAX_ITERATION);
      fprintf(stdout, "Terminated after %d iterations and %lg maximum absolute error\n", iter-1, maxerr);
    } /* For each lambda */
    fclose(ofp);
  } /* For each TTL */
  #endif
}
