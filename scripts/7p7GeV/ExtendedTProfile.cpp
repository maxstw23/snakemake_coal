#include "TObject.h"
#include "TProfile.h"
#include "./ExtendedTProfile.h"

void ExtendedTProfile::SetSumw2(int bin, float w2) {
    if (fBinEntries.fN) {
        fBinSumw2.fArray[bin] = w2;
    }
}