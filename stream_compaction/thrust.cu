#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

#include <cassert>

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            assert(n > 0);
            int* in;
            int* out;
            cudaMalloc(&in, n * sizeof(int));
            cudaMalloc(&out, n * sizeof(int));

            cudaMemcpy(in, idata, n * sizeof(int), cudaMemcpyDefault);

            auto iter = thrust::device_ptr<int>(in);
            auto out_dp = thrust::device_ptr<int>(out);
            timer().startGpuTimer();

            thrust::exclusive_scan(iter, iter + n, out_dp);

            timer().endGpuTimer();

            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDefault);

            cudaFree(in);
            cudaFree(out);
        }
    }
}
