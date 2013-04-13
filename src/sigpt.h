#ifndef __SIGPT_H
#define __SIGPT_H

typedef struct {
    float t;
    float y;
    float dy;
} sigpt_t;

/**
 * Creates an array filled with random signal points of
 * length sizeof(sigpt_t) * n.
 */
sigpt_t *sigpt_random(int seed, int n);

#endif /* __SIGPT_H */
