#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <cassert>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void naive_scan_kernel(int n, int offset, int* d_in, int* d_out)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }

            if (index >= offset)
            {
				d_out[index] = d_in[index - offset] + d_in[index];
            }
            else
            {
				d_out[index] = d_in[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            assert(n > 0);

            constexpr auto threads = 128;
            dim3 blockSize((n + threads - 1) / threads);

            int* d_in_data;
        	int* d_out_data;
			cudaMalloc(&d_in_data, n * sizeof(int));
			cudaMalloc(&d_out_data, n * sizeof(int));

            cudaMemcpy(d_in_data, idata, n * sizeof(int), cudaMemcpyDefault);

            timer().startGpuTimer();

            for (int i = 1; i < n; i *= 2)
            {
                naive_scan_kernel<<<blockSize, threads>>>(n, i, d_in_data, d_out_data);
                std::swap(d_in_data, d_out_data);
            }

            timer().endGpuTimer();

            // Exclusive
            cudaMemcpy(odata + 1, d_in_data, (n - 1) * sizeof(int), cudaMemcpyDefault);

            cudaFree(d_in_data);
            cudaFree(d_out_data);
        }
    }
}
