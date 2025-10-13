// Basic ECC field arithmetic device functions

__device__ void Add192to192(u64* r, const u64* a)
{
    asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(r[0]) : "l"(r[0]), "l"(a[0]));
    asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(r[1]) : "l"(r[1]), "l"(a[1]));
    asm volatile ("addc.u64 %0, %1, %2;" : "=l"(r[2]) : "l"(r[2]), "l"(a[2]));
}

__device__ void Sub192from192(u64* r, const u64* a)
{
    asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(r[0]) : "l"(r[0]), "l"(a[0]));
    asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(r[1]) : "l"(r[1]), "l"(a[1]));
    asm volatile ("subc.u64 %0, %1, %2;" : "=l"(r[2]) : "l"(r[2]), "l"(a[2]));
}

__device__ void AddModP(u64* r, const u64* a, const u64* b)
{
    u64 t[4];
    asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(t[0]) : "l"(a[0]), "l"(b[0]));
    asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(t[1]) : "l"(a[1]), "l"(b[1]));
    asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(t[2]) : "l"(a[2]), "l"(b[2]));
    asm volatile ("addc.u64 %0, %1, %2;" : "=l"(t[3]) : "l"(a[3]), "l"(b[3]));

    if (t[3] || t[2] >= 0xFFFFFFFEFFFFFC2F) {
        Sub192from192(t, (const u64*)&SECP256K1_P);
    }
    Copy_u64_x4(r, t);
}

__device__ void SubModP(u64* r, const u64* a, const u64* b)
{
    u64 t[4];
    asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(t[0]) : "l"(a[0]), "l"(b[0]));
    asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(t[1]) : "l"(a[1]), "l"(b[1]));
    asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(t[2]) : "l"(a[2]), "l"(b[2]));
    asm volatile ("subc.u64 %0, %1, %2;" : "=l"(t[3]) : "l"(a[3]), "l"(b[3]));

    if (t[3] >> 63) {
        Add192to192(t, (const u64*)&SECP256K1_P);
    }
    Copy_u64_x4(r, t);
}

__device__ void MulModP(u64* r, const u64* a, const u64* b)
{
    u64 t[8];
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++) {
            u64 p = __umul64hi(a[i], b[j]) + (t[i+j] >> 32) + carry;
            t[i+j] = (t[i+j] & 0xFFFFFFFF) | ((p & 0xFFFFFFFF) << 32);
            carry = p >> 32;
        }
        t[i+4] = carry;
    }

    // Fast modular reduction modulo p = 2^256 - 2^32 - 977
    u64 acc = t[7];
    acc = (acc << 32) | (t[6] >> 32);
    t[6] = (t[6] & 0xFFFFFFFF) | (t[5] << 32);
    t[5] = (t[5] >> 32) | (t[4] << 32);
    t[4] = t[4] >> 32;

    while (acc) {
        u64 carry = 0;
        carry = __umul64hi(acc, 977) + (t[0] >> 32);
        t[0] = (t[0] & 0xFFFFFFFF) | ((acc * 977) << 32);
        
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            u64 p = carry + (t[i] >> 32);
            t[i] = (t[i] & 0xFFFFFFFF) | ((p & 0xFFFFFFFF) << 32);
            carry = p >> 32;
        }
        acc = carry;
    }

    Copy_u64_x4(r, t);
}

__device__ void SqrModP(u64* r, const u64* a)
{
    MulModP(r, a, a);
}

__device__ void NegModP(u64* a)
{
    u64 t[4];
    asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(t[0]) : "l"(SECP256K1_P[0]), "l"(a[0]));
    asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(t[1]) : "l"(SECP256K1_P[1]), "l"(a[1]));
    asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(t[2]) : "l"(SECP256K1_P[2]), "l"(a[2]));
    asm volatile ("subc.u64 %0, %1, %2;" : "=l"(t[3]) : "l"(SECP256K1_P[3]), "l"(a[3]));

    Copy_u64_x4(a, t);
}

// Binary GCD for modular inverse
__device__ void InvModP(u32* r)
{
    u64 u[4] = { r[0], r[1], r[2], r[3] };
    u64 v[4] = { SECP256K1_P[0], SECP256K1_P[1], SECP256K1_P[2], SECP256K1_P[3] };
    u64 x1[4] = { 1, 0, 0, 0 };
    u64 x2[4] = { 0, 0, 0, 0 };

    while (!(u[0] == 1 && u[1] == 0 && u[2] == 0 && u[3] == 0) &&
           !(v[0] == 1 && v[1] == 0 && v[2] == 0 && v[3] == 0)) {
        while ((u[0] & 1) == 0) {
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(u[0]) : "r"(u[0]), "r"(u[1]));
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(u[1]) : "r"(u[1]), "r"(u[2]));
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(u[2]) : "r"(u[2]), "r"(u[3]));
            asm volatile ("shr.b32 %0, %1, 1;" : "=r"(u[3]) : "r"(u[3]));

            if ((x1[0] & 1) != 0) {
                bool carry = false;
                carry = __asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(x1[0]) : "l"(x1[0]), "l"(SECP256K1_P[0]));
                carry = __asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(x1[1]) : "l"(x1[1]), "l"(SECP256K1_P[1]), "l"(carry));
                carry = __asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(x1[2]) : "l"(x1[2]), "l"(SECP256K1_P[2]), "l"(carry));
                __asm volatile ("addc.u64 %0, %1, %2;" : "=l"(x1[3]) : "l"(x1[3]), "l"(SECP256K1_P[3]), "l"(carry));
            }
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(x1[0]) : "r"(x1[0]), "r"(x1[1]));
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(x1[1]) : "r"(x1[1]), "r"(x1[2]));
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(x1[2]) : "r"(x1[2]), "r"(x1[3]));
            asm volatile ("shr.b32 %0, %1, 1;" : "=r"(x1[3]) : "r"(x1[3]));
        }

        while ((v[0] & 1) == 0) {
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(v[0]) : "r"(v[0]), "r"(v[1]));
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(v[1]) : "r"(v[1]), "r"(v[2]));
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(v[2]) : "r"(v[2]), "r"(v[3]));
            asm volatile ("shr.b32 %0, %1, 1;" : "=r"(v[3]) : "r"(v[3]));

            if ((x2[0] & 1) != 0) {
                bool carry = false;
                carry = __asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(x2[0]) : "l"(x2[0]), "l"(SECP256K1_P[0]));
                carry = __asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(x2[1]) : "l"(x2[1]), "l"(SECP256K1_P[1]), "l"(carry));
                carry = __asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(x2[2]) : "l"(x2[2]), "l"(SECP256K1_P[2]), "l"(carry));
                __asm volatile ("addc.u64 %0, %1, %2;" : "=l"(x2[3]) : "l"(x2[3]), "l"(SECP256K1_P[3]), "l"(carry));
            }
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(x2[0]) : "r"(x2[0]), "r"(x2[1]));
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(x2[1]) : "r"(x2[1]), "r"(x2[2]));
            asm volatile ("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(x2[2]) : "r"(x2[2]), "r"(x2[3]));
            asm volatile ("shr.b32 %0, %1, 1;" : "=r"(x2[3]) : "r"(x2[3]));
        }

        if (u > v) {
            asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(u[0]) : "l"(u[0]), "l"(v[0]));
            asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(u[1]) : "l"(u[1]), "l"(v[1]));
            asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(u[2]) : "l"(u[2]), "l"(v[2]));
            asm volatile ("subc.u64 %0, %1, %2;" : "=l"(u[3]) : "l"(u[3]), "l"(v[3]));

            asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(x1[0]) : "l"(x1[0]), "l"(x2[0]));
            asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(x1[1]) : "l"(x1[1]), "l"(x2[1]));
            asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(x1[2]) : "l"(x1[2]), "l"(x2[2]));
            asm volatile ("subc.u64 %0, %1, %2;" : "=l"(x1[3]) : "l"(x1[3]), "l"(x2[3]));
        } else {
            asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(v[0]) : "l"(v[0]), "l"(u[0]));
            asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(v[1]) : "l"(v[1]), "l"(u[1]));
            asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(v[2]) : "l"(v[2]), "l"(u[2]));
            asm volatile ("subc.u64 %0, %1, %2;" : "=l"(v[3]) : "l"(v[3]), "l"(u[3]));

            asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(x2[0]) : "l"(x2[0]), "l"(x1[0]));
            asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(x2[1]) : "l"(x2[1]), "l"(x1[1]));
            asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(x2[2]) : "l"(x2[2]), "l"(x1[2]));
            asm volatile ("subc.u64 %0, %1, %2;" : "=l"(x2[3]) : "l"(x2[3]), "l"(x1[3]));
        }
    }
    
    r[0] = (u[0] == 1) ? x1[0] : x2[0];
    r[1] = (u[0] == 1) ? x1[1] : x2[1];
    r[2] = (u[0] == 1) ? x1[2] : x2[2];
    r[3] = (u[0] == 1) ? x1[3] : x2[3];
}