#include <cstdio>
#include "cpu.h"

#include <cassert>
#include <vector>

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            assert(n > 0);
            timer().startCpuTimer();
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = idata[i - 1] + odata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            size_t p = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
					odata[p++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return p;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            assert(n > 0);

            std::vector<int> t(n);
        	std::vector<int> s(n + 1);

            timer().startCpuTimer();

            for (int i = 0; i < n; i++)
            {
				t[i] = idata[i] != 0 ? 1 : 0;
            }

            // Inclusive scan starting at first index (to "shift" into exclusive scan)
            const auto start = s.data() + 1;
            start[0] = t[0];
            for (int i = 1; i < n; i++)
            {
                start[i] = start[i - 1] + t[i];
            }

            for (int i = 0; i < n; i++)
            {
                if (t[i] == 1)
                {
					odata[s[i]] = idata[i];
                }
			}

            timer().endCpuTimer();
            return s[n];
        }
    }
}
