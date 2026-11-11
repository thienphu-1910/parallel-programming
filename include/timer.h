#ifndef TIMER_H
#define TIMER_H

#include <chrono>

namespace panorama {
    class CpuTimer {
        public: 
            using Clock = std::chrono::high_resolution_clock;

            // Start timing when the timer is constructed
            CpuTimer()
                : start_(Clock::now()){
                }

            // Begin a new measurement
            void reset(){
                start_ = Clock::now();
            }

            // Return elapsed time without resetting the timer
            double elapsed_ms() const{
                const auto end = Clock::now();
                return std::chrono::duration<double, std::milli>(end - start_).count();
            }
        
        private:
            Clock::time_point start_;
    };

}

#endif // TIMER_H