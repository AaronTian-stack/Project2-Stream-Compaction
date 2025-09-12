#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <cassert>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upsweep(int n, int stride, int* d_in, int* d_out)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }
            int offset = stride - 1;

            // Since stride is power of two this & is like %
            // Then we check if is equal to offset: at end of group size stride
            if ((index & offset) == offset)
            {
                int p = index - stride / 2;
                d_out[index] = d_in[index] + d_in[p];
            }
            else
            {
                d_out[index] = d_in[index];
            }
        }

        __global__ void downsweep(int nodes, int stride, int* d_in, int* d_out)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= nodes)
            {
                return;
            }
            int offset = stride - 1;
            int arrayIndex = index * stride + offset;
            int leftChild = arrayIndex - stride / 2;

            d_out[leftChild] = d_in[arrayIndex];
            d_out[arrayIndex] = d_in[leftChild] + d_in[arrayIndex];
        }

        // d_in is last result
        void performSweeps(dim3 blockSize, const int threads, const int n, const int padded_n, 
            int* d_in, int* d_out)
        {
            for (int d = 0; d < ilog2ceil(n); d++)
            {
                const int stride = 1 << (d + 1);
                upsweep<<<blockSize, threads>>>(padded_n, stride, d_in, d_out);
                std::swap(d_in, d_out);
            }
            // Set last element to 0
        	cudaMemset(d_in + padded_n - 1, 0, sizeof(int));
            for (int d = 0; d < ilog2ceil(n); d++)
            {
                const int nodes = 1 << d;
                const int stride = padded_n >> d;
                downsweep<<<blockSize, threads>>>(nodes, stride, d_in, d_out);
                std::swap(d_in, d_out);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            assert(n > 0);

            constexpr auto threads = 128;
            dim3 blockSize((n + threads - 1) / threads);

            // Round up to next power of two
			const int padded_n = 1 << ilog2ceil(n);
            assert(padded_n >= n);

            int* d_in;
            int* d_out;
            
            cudaMalloc(&d_in, padded_n * sizeof(int));
            cudaMalloc(&d_out, padded_n * sizeof(int));

            cudaMemset(d_in, 0, padded_n * sizeof(int));
            cudaMemcpy(d_in, idata, n * sizeof(int), cudaMemcpyDefault);

            timer().startGpuTimer();

            performSweeps(blockSize, threads, n, padded_n, d_in, d_out);

            timer().endGpuTimer();

            cudaMemcpy(odata, d_in, n * sizeof(int), cudaMemcpyDefault);

            cudaFree(d_in);
			cudaFree(d_out);
        }

        __global__ void trueFalse(int n, const int* d_in, int* d_out)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }
            d_out[index] = d_in[index] != 0;
        }

        __global__ void scatter(int n, const int* d_in, int* d_out, const int* d_tf, const int* d_scan)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }
            if (d_tf[index] != 0)
            {
				d_out[d_scan[index]] = d_in[index];
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            assert(n > 0);

            constexpr auto threads = 128;
            dim3 blockSize((n + threads - 1) / threads);

            // Round up to next power of two
            const int padded_n = 1 << ilog2ceil(n);
            assert(padded_n >= n);

            int* d_in;
            int* d_tf;
            int* d_tf_copy;
            int* d_scan;

            cudaMalloc(&d_in, padded_n * sizeof(int));
            cudaMalloc(&d_tf, padded_n * sizeof(int));
            cudaMalloc(&d_tf_copy, padded_n * sizeof(int));
            cudaMalloc(&d_scan, padded_n * sizeof(int));

            cudaMemset(d_in, 0, padded_n * sizeof(int));
            cudaMemcpy(d_in, idata, n * sizeof(int), cudaMemcpyDefault);

            cudaMemset(d_tf, 0, padded_n * sizeof(int));

            timer().startGpuTimer();

            // t/f
            trueFalse<<<blockSize, threads>>>(n, d_in, d_tf);

            // Save original t/f
            cudaMemcpy(d_tf_copy, d_tf, padded_n * sizeof(int), cudaMemcpyDefault);

            // Scan (d_tf is actually where scan gets stored)
            performSweeps(blockSize, threads, n, padded_n, d_tf, d_scan);

            // Scatter
            // d_scan is free to use to store output
            scatter<<<blockSize, threads>>>(n, d_in, d_scan, d_tf_copy, d_tf);

            timer().endGpuTimer();

            int answer;
            cudaMemcpy(&answer, d_tf + n - 1, sizeof(int), cudaMemcpyDefault);
			// Need to add one if last element is true
			int lastElement;
			cudaMemcpy(&lastElement, d_tf_copy + n - 1, sizeof(int), cudaMemcpyDefault);
            answer += lastElement;

            cudaMemcpy(odata, d_scan, n * sizeof(int), cudaMemcpyDefault);

            cudaFree(d_in);
            cudaFree(d_tf);
            cudaFree(d_tf_copy);
            cudaFree(d_scan);

            return answer;
        }
    }
}
