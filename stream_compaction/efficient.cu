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
            const int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }
            const int offset = stride - 1;

            // Since stride is power of two this & is like %
            // Then we check if is equal to offset: at end of group size stride
            if ((index & offset) == offset)
            {
                const int p = index - stride / 2;
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
            const int offset = stride - 1;
            const int arrayIndex = index * stride + offset;
            const int leftChild = arrayIndex - stride / 2;

            d_out[leftChild] = d_in[arrayIndex];
            d_out[arrayIndex] = d_in[leftChild] + d_in[arrayIndex];
        }

        // d_in is last result
        void performSweeps(const int n, const int padded_n,
            int* d_in, int* d_out)
        {
            constexpr auto threads = 256;
            dim3 blockSize((padded_n + threads - 1) / threads);

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
                int grid = (nodes + threads - 1) / threads;
                downsweep<<<dim3(grid), threads>>>(nodes, stride, d_in, d_out);
                std::swap(d_in, d_out);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            assert(n > 0);

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

            performSweeps(n, padded_n, d_in, d_out);

            timer().endGpuTimer();

            cudaMemcpy(odata, d_in, n * sizeof(int), cudaMemcpyDefault);

            cudaFree(d_in);
            cudaFree(d_out);
        }

        __global__ void trueFalse(int n, const int* d_in, int* d_out)
        {
            const int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }
            d_out[index] = d_in[index] != 0;
        }

        __global__ void scatter(int n, const int* d_in, int* d_out, const int* d_tf, const int* d_scan)
        {
            const int index = threadIdx.x + blockIdx.x * blockDim.x;
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
        int compact(int n, int* odata, const int* idata) {
            assert(n > 0);

            constexpr auto threads = 256;
            dim3 blockSize((n + threads - 1) / threads);

            // Round up to next power of two
            const int padded_n = 1 << ilog2ceil(n);
            assert(padded_n >= n);

            int* d_in;
            int* d_tf;
            int* d_tf_copy;
            int* d_scan;

            // vidmem needs to be < 4 bytes * padded_n * 4 buffers
            // Otherwise there will be spilling (cudaMalloc will still succeed as long as the individual size is not too large)
            size_t free_bytes;
            size_t memory_needed = sizeof(int) * static_cast<size_t>(n) * 4;
            cudaMemGetInfo(&free_bytes, nullptr);
            if (free_bytes < memory_needed)
            {
                printf("GPU Memory: %.1fMB free, %.1fMB needed\n", free_bytes / (1024.0 * 1024.0), memory_needed / (1024.0 * 1024.0));
                printf("%s\n", "Not enough memory, spilling may occur");
            }

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

            cudaMemcpy(d_tf_copy, d_tf, n * sizeof(int), cudaMemcpyDefault);

            // Scan (d_tf is actually where scan gets stored)
            performSweeps(n, padded_n, d_tf, d_scan);

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

        __global__ void scan_efficient_sub(int n, int logn, int* d_in, int* d_out, bool inclusive)
        {
            const int index = threadIdx.x; // Within block
            const int block_offset = blockDim.x * blockIdx.x;

			const int globalIndex = index + block_offset;
            if (globalIndex >= n)
            {
                return;
            }

            // Do scan on subarray in shared memory
			// Num elements should be equal to threads in kernel call
            extern __shared__ int temp[];

            // Load into shared memory
			temp[index] = d_in[globalIndex];

            __syncthreads();

            // Upsweep
            for (int d = 0; d < logn; d++)
            {
                const int stride = 1 << (d + 1);
                const int offset = stride - 1;

                // Since stride is power of two this & is like %
                // Then we check if is equal to offset: at end of group size stride
                if ((index & offset) == offset)
                {
                    const int p = index - stride / 2;
                    temp[index] += temp[p];
                }
				__syncthreads();
            }

        	// Set last element to 0 based on valid elements in block
            int valid = (blockDim.x < (n - block_offset)) ? blockDim.x : (n - block_offset);
            if (index == valid - 1)
            {
                temp[index] = 0;
            }

        	// Downsweep
        	// Go in order because need to do larger strides first
            for (int d = logn - 1; d >= 0; d--)
            {
                const int stride = 1 << (d + 1);

                // Stride still power of two so can use &
				// Check if end of group
                if ((index & stride - 1) == stride - 1)
                {
                    const int leftChild = index - stride / 2;
					const int t = temp[leftChild];
					temp[leftChild] = temp[index];
					temp[index] += t;
                }
                __syncthreads();
            }
            if (inclusive)
            {
                temp[index] += d_in[globalIndex];
			}
            __syncthreads();
			d_out[globalIndex] = temp[index];
			// Do combine step in another kernel
        }

        __global__ void write_block_sums(int n, int blockSize, int* d_in, int* d_out)
        {
			const int thread = threadIdx.x + blockIdx.x * blockDim.x;
            if (thread >= n)
            {
                return;
            }
			const int globalIndex = thread * blockSize + blockSize - 1;
			d_out[thread] = d_in[globalIndex];
		}

        __global__ void add_block_increments(int n, int* d_in, int* d_out)
        {
            const int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n)
            {
                return;
            }

            const int blockIndex = index / blockDim.x;
            d_out[index] += d_in[blockIndex];
        }

		void scan_recurse(int n, const int threads,
            int* d_in, int* d_out, const int logn, bool inclusive)
        {
			assert(!(n & n - 1)); // n is power of two
            // Base case
            if (n <= threads)
            {
                scan_efficient_sub<<<1, threads, threads * sizeof(int)>>>(n, logn, d_in, d_out, inclusive);
                return;
            }

            const dim3 blockSize = dim3((n + threads - 1) / threads);
            scan_efficient_sub<<<blockSize, threads, threads * sizeof(int)>>>(n, logn, d_in, d_out, inclusive);

            int* sums;
            cudaMalloc(&sums, blockSize.x * sizeof(int));
            int* scanResult;
            cudaMalloc(&scanResult, blockSize.x * sizeof(int));

            // Extract block sums
			// One sum per block
            const dim3 extractSize = dim3((blockSize.x + threads - 1) / threads);
			write_block_sums<<<extractSize, threads>>>(blockSize.x, threads, d_out, sums);

            // Run exclusive scan on block sums
            scan_recurse(blockSize.x, threads, sums, scanResult, ilog2ceil(blockSize.x), false);

            cudaDeviceSynchronize();

            // Add block increments
			add_block_increments<<<blockSize, threads>>>(n, scanResult, d_out);


            cudaFree(sums);
            cudaFree(scanResult);
        }

        void scan_efficient(int n, int* odata, const int* idata)
        {
            assert(n > 0);

			const auto logn = ilog2ceil(n);
            const int padded_n = 1 << logn;
            assert(padded_n >= n);

            constexpr auto threads = 128;

            int* d_in;
            int* d_out;

            cudaMalloc(&d_in, padded_n * sizeof(int));
            cudaMalloc(&d_out, padded_n * sizeof(int));
            cudaMemset(d_in, 0, padded_n * sizeof(int));
            cudaMemcpy(d_in, idata, n * sizeof(int), cudaMemcpyDefault);

			timer().startGpuTimer();

            // Inclusive scan
            scan_recurse(padded_n, threads, d_in, d_out, logn, true);

            timer().endGpuTimer();

            cudaDeviceSynchronize();

			// Convert to exclusive
            odata[0] = 0;
            cudaMemcpy(odata + 1, d_out, (padded_n - 1) * sizeof(int), cudaMemcpyDefault);

            cudaFree(d_in);
            cudaFree(d_out);
        }
    }
}
