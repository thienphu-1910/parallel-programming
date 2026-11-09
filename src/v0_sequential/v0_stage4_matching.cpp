#include "v0_sequential.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace panorama {
    namespace {
        // Invalid distance is greater than number of brief bits    
        constexpr int kInvalidDistance = static_cast<int>(kBriefBits) + 1; 

        // Invalid index value for a descriptor
        constexpr std::size_t kInvalidIndex = std::numeric_limits<std::size_t>::max();

        static_assert(kBriefBits == 256,
              "Stage 4 expects BRIEF-256 descriptors.");

        static_assert(kBriefWords * 32 == kBriefBits,
              "Descriptor storage must match its bit count.");

        // Ensure 0 < match ratio < 1
        static_assert(
            kMatchRatioNumerator > 0 && kMatchRatioNumerator < kMatchRatioDenominator,
            "Matching ratio must be strictly between zero and one."
        );

        // Forward search needs two neighbors for the ratio test
        struct NearestTwo {
            std::size_t best_index = kInvalidIndex;
            int best_distance = kInvalidDistance;
            int second_distance = kInvalidDistance;
        };

        // Reverse search needs only one neighbor for cross-checking
        struct NearestOne {
            std::size_t best_index = kInvalidIndex;
            int best_distance = kInvalidDistance;
        };

        // Counts bits in groups of 2, then 4, then 8,
        // and finally adds the four byte counts.
        int popcount32(std::uint32_t value){
            value = value - ((value >> 1) & 0x55555555u); 

            value = (value & 0x33333333u)
                + ((value >> 2) & 0x33333333u);

            value = (value + (value >> 4)) & 0x0F0F0F0Fu;

            value *= 0x01010101u;

            return static_cast<int>(value >> 24);
        }


        int hamming_distance(const Descriptor& first, const Descriptor& second){
            int distance = 0;

            for (std::size_t word = 0; word < kBriefWords; word++){
                const std::uint32_t different = first.words[word] ^ second.words[word];

                distance += popcount32(different);
            }

            return distance;
        }
    }

    std::vector<Match> v0_match_descriptors(const std::vector<Descriptor>& descriptors_a,
                                            const std::vector<Descriptor>& descriptors_b){
        // Forward ratio test requires 2 candidates
        if (descriptors_a.empty() || descriptors_b.size() < 2){
            return {};
        }
        
        
        std::vector<NearestTwo> forward(descriptors_a.size());
        std::vector<NearestOne> reverse(descriptors_b.size());

        // Evaluate each pair once
        // Update both records  
        for (std::size_t i = 0; i < descriptors_a.size(); i++){
            NearestTwo& nearest_a = forward[i];

            for (std::size_t j = 0; j < descriptors_b.size(); j++){
                const int distance = hamming_distance(
                    descriptors_a[i],
                    descriptors_b[j]
                );

                // Best and second-best neighbours of A[i] in B
                if (distance < nearest_a.best_distance){
                    nearest_a.second_distance = nearest_a.best_distance;
                    nearest_a.best_distance = distance;
                    nearest_a.best_index = j;
                }
                else if (distance < nearest_a.second_distance){
                    nearest_a.second_distance = distance;
                }

                // Best neighbour of B[j] in A
                NearestOne& nearest_b = reverse[j];

                if (distance < nearest_b.best_distance) {
                    nearest_b.best_distance = distance;
                    nearest_b.best_index = i;
                }
            }
        }

        std::vector<Match> matches;
        matches.reserve(descriptors_a.size());

        // Iterating A => Deterministic match order
        for (std::size_t i = 0; i < descriptors_a.size(); i++){
            const NearestTwo& nearest = forward[i];

            if (nearest.best_index == kInvalidIndex || nearest.second_distance == kInvalidDistance){
                continue;
            }

            // d1 < 0.8 * d2, expressed using integer arithmetic:
            // 5 * d1 < 4 * d2.

            const bool passes_ratio = kMatchRatioDenominator * nearest.best_distance < kMatchRatioNumerator * nearest.second_distance;

            if (!passes_ratio) {
                continue;
            }

            const std::size_t j = nearest.best_index;

            // A[i] selected B[j]; require B[j] to select A[i].
            if (reverse[j].best_index != i) {
                continue;
            }

            matches.push_back(Match{
                i,
                j,
                nearest.best_distance
            });
        }

        return matches;
    }
}