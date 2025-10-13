#include "defs.h"
#include "utils.h"
#include <cstring>

// Structure for representing distinguished points
struct DPRecord {
    u8 d[22];  // Holds distance value
    u8 type;   // Kangaroo type (TAME, TAME2, WILD1, WILD2)
    u8 padding;
};

// Global storage for DPs
static std::vector<DPRecord> gDPTable;
static int dp_type_count[4] = {0}; // Counts per kangaroo type

// Process new distinguished points from GPU output
void CheckNewPoints() {
    bool newCollision = false;

    // Extract new DPs from GPU output buffers
    for (int i = 0; i < GpuCnt; i++) {
        u32* dp_out = GpuKangs[i]->GetDPOutput();
        int dp_cnt = dp_out[0];
        
        for (int j = 0; j < dp_cnt; j++) {
            DPRecord nrec;
            // Copy DP data from GPU output
            memcpy(&nrec, dp_out + 1 + j * sizeof(DPRecord)/4, sizeof(DPRecord));
            
            // Update statistics
            if (nrec.type < 4) {
                dp_type_count[nrec.type]++;
            }

            // Compare against existing DPs for collisions
            for (const DPRecord& pref : gDPTable) {
                // Skip same type collisions
                if (pref.type == nrec.type) continue;
                
                // Skip wild-wild collisions between same groups
                if ((pref.type >= WILD1) && (nrec.type >= WILD1)) continue;
                
                // Found matching distinguished point
                if (memcmp(pref.d, nrec.d, 22) == 0) {
                    newCollision = true;
                    
                    // Extract distances
                    EcInt w, t;
                    int TameType, WildType;
                    
                    if (pref.type != TAME) {
                        memcpy(w.data, pref.d, sizeof(pref.d));
                        if (pref.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                        memcpy(t.data, nrec.d, sizeof(nrec.d));
                        if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                        TameType = nrec.type;
                        WildType = pref.type;
                    } else {
                        memcpy(w.data, nrec.d, sizeof(nrec.d));
                        if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
                        memcpy(t.data, pref.d, sizeof(pref.d));
                        if (pref.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
                        TameType = TAME;
                        WildType = nrec.type;
                    }

                    // Try collision with both positive and negative solutions
                    if (Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) || 
                        Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true)) {
                        gSolved = true;
                        return;
                    }
                }
            }
            
            // Add new DP to table
            gDPTable.push_back(nrec);
        }
        
        // Reset GPU buffer count
        dp_out[0] = 0;
    }
}

// Verify potential solution found from collision
bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg) {
    EcPoint wild, tame;
    
    // Compute tame kangaroo point
    if (TameType == TAME) {
        tame = Pnt_HalfRange;
        tame.Multiply(t);
    } else {
        tame = Pnt_NegHalfRange; 
        tame.Multiply(t);
    }
    
    // Compute wild kangaroo point 
    if (IsNeg) {
        wild = pnt;
        wild.Negate();
    } else {
        wild = pnt;
    }
    
    EcPoint temp = wild;
    if (WildType == WILD1 || WildType == WILD2) {
        temp.Multiply(w);
    }
    
    // Compare points
    if (tame.Equals(temp)) {
        // Found valid solution - calculate discrete log
        EcInt sol = w;
        if (TameType == TAME2 || WildType >= WILD1) {
            sol.Subtract(t);
        } else {
            sol.Add(t); 
        }
        
        if (IsNeg) {
            sol.Negate();
        }
        
        // Store solution and return success
        memcpy(gSolution.data, sol.data, 40);
        return true;
    }
    
    return false;
}