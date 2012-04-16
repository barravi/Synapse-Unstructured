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
extern double pow();

extern int MAX_DEGREE;
extern int MAX_SYNAPSE;
extern int MAX_TTL;

extern double zmed;

extern int *idpk;
extern double *pk;
extern int *idsk;
extern double *sk;
extern int *idpf;
extern double *pf;

extern double alpha;

extern double (*MSG[7])(double);
extern double (*HIT[7])(double);

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double G0(double z)
{
  int k;
  double s = 0.0;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init G0\n");
  #endif
  for (k = 1; k <= MAX_DEGREE; k++)
  {
    #ifdef DEBUG
    fprintf(stdout,"Summing %lg * %d to %lg\n",pk[k],idpk[k],s);
    #endif
    s += pk[k] * pow(z,(double)idpk[k]);
  }
  #ifdef DEBUG
  fprintf(stdout,"G0(%lg)=%lg\n",z,s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End G0\n");
  #endif
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double G1(double z)
{
  int k;
  double s = 0.0;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init G1\n");
  #endif
  for (k = 1; k <= MAX_DEGREE; k++)
  {
    if(idpk[k] > 0)
    {
      s += idpk[k] * pk[k] * pow(z,(double)(idpk[k]-1));
    }
  }
  s/=pk[0];
  #ifdef DEBUG
  fprintf(stdout,"G1(%lg)=%lg\n",z,s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End G1\n");
  #endif
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double F(double z)
{
  int k;
  double s = 0.0;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init F\n");
  #endif
  for (k = 1; k <= MAX_SYNAPSE; k++)
  {
    s += sk[k] * pow(z,(double)idsk[k]);
  }
  #ifdef DEBUG
  fprintf(stdout,"F(%lg)=%lg\n",z,s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End F\n");
  #endif
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double M(double z)
{
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init M\n");
  #endif
  #ifdef DEBUG
  fprintf(stdout,"M(%lg)=%lg\n",z,F(G0(z)));
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End M\n");
  #endif
  return F(G0(z));
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double N(double z)
{
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init N\n");
  #endif
  #ifdef DEBUG
  fprintf(stdout,"N(%lg)=%lg\n",z,G1(z)/G0(z)*F(G0(z)));
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End N\n");
  #endif
  return (G1(z) / G0(z) * F(G0(z)));
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double Q(double z)
{
  int k;
  double s = 0.0;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init Q\n");
  #endif
  for(k = 1; k <= MAX_SYNAPSE; k++)
  {
    s+=sk[k]*pow(G0(1.0+pf[k]*(z-1.0)),(double)idsk[k]);
  }
  #ifdef DEBUG
  fprintf(stdout,"Q(%lg)=%lg\n",z,s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End Q\n");
  #endif
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double R(double z)
{
  int k;
  double s = 0.0;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init R\n");
  #endif
  for(k = 1; k <= MAX_SYNAPSE; k++)
  {
    s+=sk[k]*pow(G0(1.0+pf[k]*(z-1.0)),(double)(idsk[k]-1))*G1(1.0+pf[k]*(z-1.0));
  }
  #ifdef DEBUG
  fprintf(stdout,"R(%lg)=%lg\n",z,s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End R\n");
  #endif
  if (0) // TODO: Change with a flag for limited/unlimited behaviour!
  {
    s = alpha + (1.0 - alpha) * s;
  }
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double Q1(double z)
{
  return (Q(z));
}
double Q2(double z)
{
  return (Q(R(z)));
}
double Q3(double z)
{
  return (Q(R(R(z))));
}
double Q4(double z)
{
  return (Q(R(R(R(z)))));
}
double Q5(double z)
{
  return (Q(R(R(R(R(z))))));
}
double Q6(double z)
{
  return (Q(R(R(R(R(R(z)))))));
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double T(double z)
{
  int d;
  double s = 1;
  for (d = 1; d <= MAX_TTL; d++)
  {
    s *=  MSG[d](z);
  }
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double H1(double z)
{
  return (Q1(1.0 + alpha * (z-1.0)));
}
double H2(double z)
{
  return (Q2(1.0 + alpha * (z-1.0)));
}
double H3(double z)
{
  return (Q3(1.0 + alpha * (z-1.0)));
}
double H4(double z)
{
  return (Q4(1.0 + alpha * (z-1.0)));
}
double H5(double z)
{
  return (Q5(1.0 + alpha * (z-1.0)));
}
double H6(double z)
{
  return (Q6(1.0 + alpha * (z-1.0)));
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double H(double z)
{
  int d;
  double s = 1;
  for(d = 1; d <= MAX_TTL; d++)
  {
    s *=  HIT[d](z);
  }
  return s;
}

//*****************
#ifdef DAFINIRE
/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double NR(double x, int d, int k)
{
  double s;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init NR\n");
  #endif
  s =  pow((1 + (x-1)*pr[k][d]),id[k]);
  #ifdef DEBUG
  //fprintf(stdout,"NR(%lg,%d,%d)=%lg\n",x,d,id[k],s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End NR\n");
  #endif
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double R(double x, int d)
{
  int k;
  double s;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init R\n");
  #endif
  s = 0.0;
  for(k = 1; k <= MAX_DEGREE; k++)
  {
    s +=  pk[k] * pow((1 + (x-1)*pr[k][d]),id[k]);
  }
  #ifdef DEBUG
  //fprintf(stdout,"R(%lg,%d)=%lg\n",x,d,s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End R\n");
  #endif
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double S(double x, int d)
{
  int k;
  double s;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init S\n");
  #endif
  s = 0.0;
  for (k = 1; k <= MAX_DEGREE; k++)
  {
    if (rfw == TRUE)
    {
      s += id[k] * pk[k] * (1 + ( pow(1 + (x - 1) * pr[k][d], id[k] - 1) -1) * a[k] * pf[k][d]);
    }
    else
    {
      s += id[k] * pk[k] * (1 + ( pow(1 + (x - 1) * pr[k][d], id[k] - 1) -1) * a[k] * pf[k][d] * ( 1 - gamma_[k]));
    }
  }
  s /= zmed;
  #ifdef DEBUG
  //fprintf(stdout,"S(%lg,%d)=%lg\n",x,d,s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End S\n");
  #endif
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double M(double x)
{
  int k;
  double s;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init M\n");
  #endif
  s = 0.0;
  for (k = 1; k <= MAX_DEGREE; k++)
  {
    s += (1.0 - gamma_[k]) * pk[k] * pow((1 + (x-1)*pf[k][0]),id[k]);
  }
  #ifdef DEBUG
  //fprintf(stdout,"M(%lg)=%lg\n",x,s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End M\n");
  #endif
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double N(double x, int d)
{
  int k;
  double s;
  #ifdef DEBUG_CALLS
  fprintf(stdout,"Init N\n");
  #endif
  s = 0.0;
  for (k = 1; k <= MAX_DEGREE; k++)
  {
    if(rfw == TRUE)
    {
      s += id[k] * pk[k] * (1 + ( pow(1 + (x - 1) * pf[k][d], id[k] - 1) -1) * pr[k][d] * a[k]);
    }
    else
    {
      s += id[k] * pk[k] * (1 + ( pow(1 + (x - 1) * pf[k][d], id[k] - 1) -1) * pr[k][d] * a[k] * ( 1 - gamma_[k]));
    }
  }
  s /= zmed;
  #ifdef DEBUG
  //fprintf(stdout,"N(%lg,%d)=%lg\n",x,d,s);
  #endif
  #ifdef DEBUG_CALLS
  fprintf(stdout,"End N\n");
  #endif
  return s;
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double HIT1(double x)
{
  return (M(1+(x-1) * gamma_med[1]) / QO(1));
}
double HIT2(double x)
{
  return (M(N(1+(x-1) * gamma_med[2],1)) / QO(1));
}
double HIT3(double x)
{
  return (M(N(N(1+(x-1) * gamma_med[3],2),1)) / QO(1));
}
double HIT4(double x)
{
  return (M(N(N(N(1+(x-1) * gamma_med[4],3),2),1)) /QO(1));
}
double HIT5(double x)
{
  return (M(N(N(N(N(1+(x-1) * gamma_med[5],4),3),2),1)) / QO(1));
}
double HIT6(double x)
{
  return (M(N(N(N(N(N(1+(x-1) * gamma_med[6],5),4),3),2),1)) / QO(1));
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double QM1(double x)
{
  return (M(1+(x-1) * pr_med[1]) / QO(1));
}
double QM2(double x)
{
  return (M(N(1+(x-1) * pr_med[2],1)) / QO(1));
}
double QM3(double x)
{
  return (M(N(N(1+(x-1) * pr_med[3],2),1)) / QO(1));
}
double QM4(double x)
{
  return (M(N(N(N(1+(x-1) * pr_med[4],3),2),1)) /QO(1));
}
double QM5(double x)
{
  return (M(N(N(N(N(1+(x-1) * pr_med[5],4),3),2),1)) / QO(1));
}
double QM6(double x)
{
  return (M(N(N(N(N(N(1+(x-1) * pr_med[6],5),4),3),2),1)) / QO(1));
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double QS1(double x)
{
  return (R(1+(x-1) * pf_med[0],1));
}
double QS2(double x)
{
  return (R(S(1+(x-1) * pf_med[0],1),2));
}
double QS3(double x)
{
  return (R(S(S(1+(x-1) * pf_med[0],1),2),3));
}
double QS4(double x)
{
  return (R(S(S(S(1+(x-1) * pf_med[0],1),2),3),4));
}
double QS5(double x)
{
  return (R(S(S(S(S(1+(x-1) * pf_med[0],1),2),3),4),5));
}
double QS6(double x)
{
  return (R(S(S(S(S(S(1+(x-1) * pf_med[0],1),2),3),4),5),6));
}

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
double NSRC1(double x, int k)
{
  return(NR(1+(x-1)*pf_med[0],1,k));
}
double NSRC2(double x, int k)
{
  return(NR(S(1+(x-1)*pf_med[0],1),2,k));
}
double NSRC3(double x, int k)
{
  return(NR(S(S(1+(x-1)*pf_med[0],1),2),3,k));
}
double NSRC4(double x, int k)
{
  return(NR(S(S(S(1+(x-1)*pf_med[0],1),2),3),4,k));
}
double NSRC5(double x, int k)
{
  return(NR(S(S(S(S(1+(x-1)*pf_med[0],1),2),3),4),5,k));
}
double NSRC6(double x, int k)
{
  return(NR(S(S(S(S(S(1+(x-1)*pf_med[0],1),2),3),4),5),6,k));
}
#endif //#ifdef DAFINIRE
