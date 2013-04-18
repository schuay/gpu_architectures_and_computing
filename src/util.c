#include "util.h"

#include <stdlib.h>
#include <stdio.h>


float *random_array(int seed, int length)
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




sigpt_t *read_signal_file(const char* filename) {

	FILE *fh;
	sigpt_t *signal;

	fh = fopen(filename, "r");
	if (fh) {
		char line[1024];
		int line_count = 0;
		while (fgets(line, 1024, fh))
			line_count++;

		fseek(fh, 0, SEEK_SET);

		signal = (sigpt_t*) malloc(line_count * sizeof(sigpt_t));
		if (signal == NULL)
			return NULL;

		int i = 0;
		while (fgets(line, 1024, fh) || i > line_count) {
			signal[i].t = signal[i].y = signal[i].dy = 0;
			sscanf(line, "%f %f %f", &signal[i].t, &signal[i].y, &signal[i].dy);
			i++;
		}

		fclose(fh);

		return signal;
	} else
		return NULL;
}


int write_signal_file(const char* filename, const sigpt_t* signal, int n) {

	FILE *fh;

	fh = fopen(filename, "w+");
	if (fh) {
		for (int i = 0; i < n; i++) {
			fprintf(fh, "%.10f %.10f %.10f\n", signal[i].t, signal[i].y, signal[i].dy);
		}
		fclose(fh);
	}
	return 0;
}
