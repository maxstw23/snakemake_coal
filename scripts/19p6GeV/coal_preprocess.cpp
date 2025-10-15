#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "Math/SpecFunc.h"
#include "TSystem.h"
#include "TF1.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TProfile.h"
#include "TMath.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TList.h"
#include "TString.h"
#include "TRandom.h"
#include "TRandom3.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TVector3.h"
#include <algorithm>
#include <array>
#include <stdlib.h>
#include "TComplex.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TVector2.h"
#include "TDatabasePDG.h"
#include "TParticlePDG.h"
#include "TEfficiency.h"
#include <string>
#include <sstream>
#include <string>
#include <limits>


void coal_preprocess(TString inFileName = "./result9.root", float y_cut = 0.6)
{
    TFile *f = new TFile(inFileName.Data(), "READ");
    const char* basename = gSystem->BaseName(inFileName);
    const char* dirname = gSystem->DirName(inFileName);
    TFile *fout = new TFile(Form("%s/preprocessed_%s", dirname, basename), "RECREATE");

    const char* particles[6] = {"piplus", "piminus", "kplus", "kminus", "proton", "antiproton"};
    const char* EP_methods[2] = {"TPC", "EPD"};

    for (int par=0; par<6; par++)
    {
        const char* particle = particles[par];
        for (int ep=0; ep<2; ep++)
        {
            for (int i=0; i<9; i++)
            {
                const char* EP_method = EP_methods[ep];
                std::string par_str;
                if (!strcmp(particle, "piplus")) par_str = "pip";
                if (!strcmp(particle, "piminus")) par_str = "pim";
                if (!strcmp(particle, "kplus")) par_str = "Kp";
                if (!strcmp(particle, "kminus")) par_str = "Km";
                if (!strcmp(particle, "proton")) par_str = "P";
                if (!strcmp(particle, "antiproton")) par_str = "AP";

                TProfile2D *p2D = (TProfile2D*)f->Get(Form("h%s_%s_v2_y_pt_%d", particle, EP_method, i+1));
                int y_lo_bin = p2D->GetXaxis()->FindBin(-y_cut);
                int y_hi_bin = p2D->GetXaxis()->FindBin(y_cut);
                TProfile *p = p2D->ProfileY(Form("h%s_%s_v2_pt_%d", particle, EP_method, i+1), y_lo_bin, y_hi_bin);
                p->Write();
            }
        }
    }

    // store the TOF stuff
    for (int i = 0; i < 9; i++)
    {
        TH2D* hgpTeta = (TH2D*)f->Get(Form("hgpTeta_%d", i+1))->Clone(Form("hgpTeta_%d", i+1));
        TH2D* hgpTeta_TOF = (TH2D*)f->Get(Form("hgpTeta_TOF_%d", i+1))->Clone(Form("hgpTeta_TOF_%d", i+1));
        hgpTeta->Write();
        hgpTeta_TOF->Write();
    }

    // store resolution
    TProfile *hTPC_res = (TProfile*)f->Get("hTPCEP_ew_cos")->Clone("hTPCEP_ew_cos");
    hTPC_res->Write();
    TProfile *hEPD_res_1 = (TProfile*)f->Get("hEPDEP_ew_cos_1")->Clone("hEPDEP_ew_cos_1");
    hEPD_res_1->Write();
    TProfile *hEPD_res_2 = (TProfile*)f->Get("hEPDEP_ew_cos_2")->Clone("hEPDEP_ew_cos_2");
    hEPD_res_2->Write();

    fout->Close();
    f->Close();
}