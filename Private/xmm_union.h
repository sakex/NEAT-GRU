//
// Created by alexandre on 22.06.20.
//

#ifndef NEAT_GRU_XMM_UNION_H
#define NEAT_GRU_XMM_UNION_H

#include <xmmintrin.h>

union alignas(16) xmm {
    __m128 simd;
    double data[4];
};

#endif //NEAT_GRU_XMM_UNION_H
