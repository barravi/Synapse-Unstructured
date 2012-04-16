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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../INCLUDES/const.h"

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
char *emalloc ( unsigned sz )
{/* Init emalloc */
  char * callptr;
  if ((callptr = malloc(sz)) == NULL)
  { /* Error */
    fprintf(stdout, "Runtime Error: emalloc: couldn't fill request for %d\n", sz);
    exit(1);
  } /* Error */
  return(callptr);
} /* End emalloc */

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
void *ecalloc ( size_t nitm, size_t sz, char func[MAXSTRING], char arr[MAXSTRING] )
{ /* Init ecalloc */
  void * callptr;
  if ((callptr = calloc(nitm, sz)) == NULL)
  { /* Error */
    fprintf(stdout, "\nRuntime Error : ecalloc couldn't fill request for array %s\nfrom function %s (N=%d, Size=%d)\n", arr, func, (int)nitm, (int)sz);
    exit(1);
  } /* Error */
  return(callptr);
} /* End ecalloc */

/**************************************************************/
/* NAME : */
/* DESCRIPTION : */
/* PARAMETERS : */
/* RETURN VALUE : */
/**************************************************************/
FILE *efopen ( char file[MAXSTRING], char mode[5] )
{ /* Init efopen */
  extern FILE *fopen();
  FILE *callptr;
  if ((callptr = fopen(file, mode)) == NULL)
  { /* Error */
    fprintf(stdout,"Runtime Error: fopen couldn't open %s, in mode %s\n",file, mode );
    exit(1);
  } /* Error */
  return(callptr);
} /* End efopen */
