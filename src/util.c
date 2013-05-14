#include "util.h"

#include <stdlib.h>
#include <stdio.h>

float *
random_array(int seed,
             int length)
{
    float *fs = (float *)malloc(length * sizeof(float));
    if (fs == NULL) {
        return fs;
    }

    srand(seed);

    for (int i = 0; i < length; i++) {
        fs[i] = (float)rand() / (float)RAND_MAX;
    }

    return fs;
}

int
read_signal_file(const char *filename,
                 sigpt_t **signal)
{
    FILE *fh;
    sigpt_t *s;

    fh = fopen(filename, "r");
    if (fh == NULL) {
        return -1;
    }

    char line[1024];
    int line_count = 0;
    while (fgets(line, 1024, fh)) {
        line_count++;
    }

    fseek(fh, 0, SEEK_SET);

    s = (sigpt_t*) malloc(line_count * sizeof(sigpt_t));
    *signal = s;
    if (s == NULL) {
        return -1;
    }

    int i = 0;
    while (fgets(line, 1024, fh) || i > line_count) {
        s[i].t = s[i].y = s[i].dy = 0;
        sscanf(line, "%f %f %f", &s[i].t, &s[i].y, &s[i].dy);
        i++;
    }

    fclose(fh);

    return line_count;
}

int
write_signal_file(const char* filename,
                  const sigpt_t* signal,
                  int n)
{
    FILE *fh;

    fh = fopen(filename, "w+");
    if (fh == NULL) {
        return -1;
    }

    for (int i = 0; i < n; i++) {
        fprintf(fh, "%.10f %.10f %.10f\n", signal[i].t, signal[i].y, signal[i].dy);
    }
    fclose(fh);

    return 0;
}
