#ifndef ExtendedTProfile_H
#define ExtendedTProfile_H

#include "TObject.h"
#include "TProfile.h"

class ExtendedTProfile : public TProfile
{
public:
    ExtendedTProfile() : TProfile() {}
    ExtendedTProfile(const TProfile& p) : TProfile(p) {}
    ExtendedTProfile(const char* name, const char* title, Int_t nbinsx, Double_t xlow, Double_t xup, Double_t ylow, Double_t yup, Option_t* option = "") 
        : TProfile(name, title, nbinsx, xlow, xup, ylow, yup, option) {}
    virtual ~ExtendedTProfile() {}

    void SetSumw2(int nbin, float w2);

ClassDef(ExtendedTProfile,1)
};

#endif