// GLV endomorphism and batch operations for RCKangaroo
#pragma once

#include "defs.h"

// GLV splitting function
__device__ void split_scalar_glv(u64* k, u64* k1, u64* k2) {
    // Split k into k1 and k2 using GLV lambda values
    // k = k1 + k2*lambda (mod n)
    u64 temp[4];
    mul_mod_n(k2, k, glv_lambda1);
    mul_mod_n(temp, k2, glv_lambda2);
    sub_mod_n(k1, k, temp);
}

// Point multiplication using GLV method
__device__ void point_mul_glv(u64* rx, u64* ry, const u64* px, const u64* py, const u64* k) {
    u64 k1[4], k2[4];
    u64 qx[4], qy[4];
    
    // Split scalar k
    split_scalar_glv(k, k1, k2);
    
    // Calculate endomorphism point Q = φ(P)
    mul_mod_p(qx, px, glv_beta);
    copy_u64(qy, py, 4);
    
    // Simultaneous double-and-add using GLV
    u64 rx1[4], ry1[4], rx2[4], ry2[4];
    point_mul_base(rx1, ry1, px, py, k1);
    point_mul_base(rx2, ry2, qx, qy, k2);
    point_add(rx, ry, rx1, ry1, rx2, ry2);
}

// Batch inversion using Montgomery's trick
__device__ void batch_invert(u64* inputs[], int count, u64* outputs[]) {
    if (count == 0) return;
    
    // Step 1: Compute running products
    u64* products[BATCH_SIZE];
    copy_u64(products[0], inputs[0], 4);
    
    for (int i = 1; i < count; i++) {
        mul_mod_p(products[i], products[i-1], inputs[i]);
    }
    
    // Step 2: Invert the final product
    u64 inv[4];
    mod_inverse(inv, products[count-1]);
    
    // Step 3: Walk backwards to get individual inverses
    for (int i = count-1; i > 0; i--) {
        mul_mod_p(outputs[i], inv, products[i-1]);
        mul_mod_p(inv, inv, inputs[i]);
    }
    copy_u64(outputs[0], inv, 4);
}