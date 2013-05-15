#ifndef __GLOBALS_H
#define __GLOBALS_H

#define NBLOCKS (256)
#define NTHREADS (256)

#define FLOAT_DELTA (1e-10f)

#define CUDA_MAX(a, b) (((a) > (b)) * (a) + ((a) <= (b)) * (b))
#define CUDA_MIN(a, b) (((a) < (b)) * (a) + ((a) >= (b)) * (b))

#endif /* __GLOBALS_H */
