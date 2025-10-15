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
#include "./ExtendedTProfile.h"
#include <string>
#include "./Efficiency.h"

R__LOAD_LIBRARY(./ExtendedTProfile_cpp.so)
R__LOAD_LIBRARY(./Efficiency_cpp.so)

using namespace std;

const float PI = TMath::Pi();
const float pion_mass = 0.13957;
const float proton_mass = 0.93827;
// const int energy = 14;

void v2_eff_correction_1D(const char* rawFileName, const char* outFileName, float y_cut, float pT_lo_nq, float pT_hi_nq, const char* EPD_method, bool use_mT=false)
{   
    char *inFileName = Form("%s", rawFileName);
    TFile *f = new TFile(inFileName, "READ");

    // generate TOF efficiency
    gSystem->Exec(Form("root -l -b -q 'draw_TOF_eff_2.cpp(\"%s\", \"TOFEfficiency.root\", %f)'", inFileName, y_cut));
    TFile *feff = new TFile("./TOFEfficiency.root", "READ");

    //const char* particle = "piminus"; // "pipluslus", "piminus", "kplus", "kminus", "proton", "antiproton"
    //const char* EP_method = "TPC"; // "TPC", "EPD"
    const char* particles[3] = {"piplus", "piminus", "antiproton"};
    const char* EP_methods[2] = {"TPC", "EPD"};

    float v2[3][2][9] = {0};
    float v2_err[3][2][9] = {0};
    float v2_counts[3][9] = {0};

    for (int par=0; par<3; par++)
    {   
        const char* particle = particles[par];
        for (int ep=0; ep<2; ep++)
        {
            const char* EP_method = EP_methods[ep];
            float pT_min = 0.2;
            float pT_max = 2.0;
            float pT_TOFth = 0.2;
            if (!strcmp(particle, "piplus") || !strcmp(particle, "piminus")) {pT_TOFth = 0.4; pT_min = pT_lo_nq*2; pT_max = pT_hi_nq*2;} // 0.105 - 1.86 mT range for 0.2 - 2.0 pT
            if (!strcmp(particle, "kplus") || !strcmp(particle, "kminus")) pT_TOFth = 0.4;
            if (!strcmp(particle, "proton") || !strcmp(particle, "antiproton")) {pT_TOFth = 0.6; pT_min = pT_lo_nq*3; pT_max = pT_hi_nq*3;} // 0.03 - 1.27 mT range for 0.2 - 2.0 pT
            if (use_mT)
            {
                if (!strcmp(particle, "piplus") || !strcmp(particle, "piminus")) 
                {
                    pT_min = sqrt((pT_min+pion_mass) * (pT_min+pion_mass) - pion_mass * pion_mass);
                    pT_max = sqrt((pT_max+pion_mass) * (pT_max+pion_mass) - pion_mass * pion_mass);
                }
                if (!strcmp(particle, "proton") || !strcmp(particle, "antiproton")) 
                {
                    pT_min = sqrt((pT_min+proton_mass) * (pT_min+proton_mass) - proton_mass * proton_mass);
                    pT_max = sqrt((pT_max+proton_mass) * (pT_max+proton_mass) - proton_mass * proton_mass);
                }
            }
            // assert(pT_min < pT_TOFth && "Check particle type!");
            // vector<float> P0, P1, P2;
            // if (!strcmp(particle, "piplus")) {P0 = P0_pip; P1 = P1_pip; P2 = P2_pip;}
            // if (!strcmp(particle, "piminus")) {P0 = P0_pim; P1 = P1_pim; P2 = P2_pim;}
            // if (!strcmp(particle, "kplus")) {P0 = P0_K; P1 = P1_K; P2 = P2_K;}
            // if (!strcmp(particle, "kminus")) {P0 = P0_K; P1 = P1_K; P2 = P2_K;}
            // if (!strcmp(particle, "proton")) {P0 = P0_P; P1 = P1_P; P2 = P2_P;}
            // if (!strcmp(particle, "antiproton")) {P0 = P0_AP; P1 = P1_AP; P2 = P2_AP;}
            std::string par_str;
            if (!strcmp(particle, "piplus")) par_str = "pip";
            if (!strcmp(particle, "piminus")) par_str = "pim";
            if (!strcmp(particle, "kplus")) par_str = "Kp";
            if (!strcmp(particle, "kminus")) par_str = "Km";
            if (!strcmp(particle, "proton")) par_str = "P";
            if (!strcmp(particle, "antiproton")) par_str = "AP";

            Efficiency *eff = new Efficiency();
            for (int cen=1; cen<=9; cen++)
            {
                TProfile *p = (TProfile*)f->Get(Form("h%s_%s_v2_pt_%d", particle, EP_method, cen));
                //TH1D *hpT = (TH1D*)f->Get(Form("hgpT_%d", cen));
                //TH1D *hpT_TOF = (TH1D*)f->Get(Form("hgpT_TOF_%d", cen));
                //TEfficiency *hTOF_Eff = new TEfficiency(*hpT_TOF, *hpT);
                TEfficiency *hTOF_Eff = (TEfficiency*)feff->Get(Form("hTOFEff_%d", cen));
                ExtendedTProfile *hv2_pt = new ExtendedTProfile(*p); 
                hv2_pt->Sumw2();
                hv2_pt->SetErrorOption("s");

                // efficiency correction
                for (int i = 0; i < hv2_pt->GetNbinsX(); i++)
                {
                    float pt = hv2_pt->GetBinCenter(i + 1);
                    if (pt < pT_min || pt > pT_max) continue;
                    float TOF_eff = 1.0;
                    if (pt > pT_TOFth) TOF_eff = hTOF_Eff->GetEfficiency(hTOF_Eff->FindFixBin(pt));
                    float TPC_eff = eff->GetEfficiency1D(pt, cen, par_str); //P0[cen-1]*exp(-pow(P1[cen-1]/pt,P2[cen-1]));
                    float weight = 1.0 /TOF_eff / TPC_eff;
                    float content = hv2_pt->GetBinContent(i + 1);
                    float error = hv2_pt->GetBinError(i + 1);
                    float entries = hv2_pt->GetBinEntries(i + 1);
                    float entries_eff = hv2_pt->GetBinEffectiveEntries(i + 1);
                    float sumw2 = hv2_pt->GetBinSumw2()->At(i + 1);

                    float W = entries * weight;           // set W(j)
                    float H = content * entries * weight; // set H(j), which is h(j) * W(j)
                    float sumw2_new = sumw2 * weight * weight;
                    float entries_eff_new = W * W / sumw2_new;
                    float E = (error * error + content * content) * entries * weight; // set E(j), sum of weights squared

                    hv2_pt->SetBinEntries(i + 1, W); // set W(j)
                    hv2_pt->SetBinContent(i + 1, H); // set H(j), which is h(j) * W(j)
                    float entries_eff_current = hv2_pt->GetBinEffectiveEntries(i + 1);
                    hv2_pt->SetBinError(i + 1, sqrt(E)); // set sqrt(E(j))
                    hv2_pt->SetSumw2(i + 1, sumw2_new);
                }

                float cut1 = hv2_pt->GetXaxis()->GetBinLowEdge(1);
                float cut2 = hv2_pt->GetXaxis()->GetBinLowEdge(hv2_pt->GetXaxis()->FindFixBin(pT_min));
                float cut3 = hv2_pt->GetXaxis()->GetBinLowEdge(hv2_pt->GetXaxis()->FindFixBin(pT_max));
                float cut4 = hv2_pt->GetXaxis()->GetBinUpEdge(hv2_pt->GetXaxis()->FindFixBin(pT_max));
                Double_t cuts[4] = {cut1, cut2, cut3, cut4};
                TProfile* hv2_pt_new = (TProfile*)hv2_pt->Rebin(3, "hv2_pt_new", cuts);
                hv2_pt_new->SetErrorOption("");
                v2[par][ep][cen-1] = hv2_pt_new->GetBinContent(2);
                v2_err[par][ep][cen-1] = hv2_pt_new->GetBinError(2);
                v2_counts[par][cen-1] = hv2_pt_new->GetBinEffectiveEntries(2);
            }  
        }
    }
    // pretty print v2 and v2 err to csv
    ofstream outputFile(outFileName);
    outputFile << "piplus_counts,piminus_counts,antiproton_counts,";
    outputFile << "piplus_v2_TPC,piminus_v2_TPC,antiproton_v2_TPC,";
    outputFile << "piplus_v2_err_TPC,piminus_v2_err_TPC,antiproton_v2_err_TPC,";
    outputFile << "piplus_v2_EPD,piminus_v2_EPD,antiproton_v2_EPD,";
    outputFile << "piplus_v2_err_EPD,piminus_v2_err_EPD,antiproton_v2_err_EPD" << endl;
    for (int i=0; i<9; i++)
    {
        outputFile << v2_counts[0][i] << ",";
        outputFile << v2_counts[1][i] << ",";
        outputFile << v2_counts[2][i] << ",";
        outputFile << v2[0][0][i] << ",";
        outputFile << v2[1][0][i] << ",";
        outputFile << v2[2][0][i] << ",";
        outputFile << v2_err[0][0][i] << ",";
        outputFile << v2_err[1][0][i] << ",";
        outputFile << v2_err[2][0][i] << ",";
        outputFile << v2[0][1][i] << ",";
        outputFile << v2[1][1][i] << ",";
        outputFile << v2[2][1][i] << ",";
        outputFile << v2_err[0][1][i] << ",";
        outputFile << v2_err[1][1][i] << ",";
        outputFile << v2_err[2][1][i] << endl;
    }

    outputFile.close();
    f->Close();
    feff->Close();
}
