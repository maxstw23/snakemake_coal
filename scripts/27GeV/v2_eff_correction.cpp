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

#define num_par 6
#define num_EP 2
#define num_cen 9
#define num_pt_bins 500

R__LOAD_LIBRARY(./ExtendedTProfile_cpp.so)
R__LOAD_LIBRARY(./Efficiency_cpp.so)

using namespace std;

const float PI = TMath::Pi();
const float pion_mass = 0.13957;
const float proton_mass = 0.93827;
const float kion_mass = 0.49367;
// const int energy = 14;

void v2_eff_correction(const char* rawFileName, const char* outFileName, float y_cut, float pT_lo_nq, float pT_hi_nq, const char* EPD_method, bool use_2D, bool use_mT=false, int sys_tag=0)
{
    // preprocess the input file in case of 2D analysis
    char *inFileName = Form("%s", rawFileName);
    if (use_2D)
    {
        gSystem->Exec(Form("root -l -b -q 'coal_preprocess.cpp(\"%s\", %f)'", rawFileName, y_cut));
        const char* basename = gSystem->BaseName(rawFileName);
        const char* dirname = gSystem->DirName(rawFileName);
        inFileName = Form("%s/preprocessed_%s", dirname, basename);
    }
    TFile *f = new TFile(inFileName, "READ");

    // read resolution
    float res[num_EP][num_cen] = {0};
    float res_err[num_EP][num_cen] = {0};
    TProfile *hTPC_res = (TProfile*)f->Get("hTPCEP_ew_cos");
    TProfile *hEPD_res;
    if (!strcmp(EPD_method, "1st")) hEPD_res = (TProfile*)f->Get("hEPDEP_ew_cos_1");
    else if (!strcmp(EPD_method, "2nd")) hEPD_res = (TProfile*)f->Get("hEPDEP_ew_cos_2");
    else throw invalid_argument("Invalid EPD method!");
    for (int i=0; i<num_cen; i++)
    {
        res[0][i] = sqrt(hTPC_res->GetBinContent(i+1));
        res_err[0][i] = hTPC_res->GetBinError(i+1) / (2*res[0][i]);
        if (!strcmp(EPD_method, "1st"))
        {
            res[1][i] = hEPD_res->GetBinContent(i+1);
            res_err[1][i] = hEPD_res->GetBinError(i+1);
        }
        else if (!strcmp(EPD_method, "2nd"))
        {
            res[1][i] = sqrt(hEPD_res->GetBinContent(i+1));
            res_err[1][i] = hEPD_res->GetBinError(i+1) / (2*res[1][i]);
        }
    }
    std::string outFileNameStr(outFileName);
    ofstream resFile(outFileNameStr.replace(outFileNameStr.find(".csv"), 4, "_res.csv"));
    resFile << "TPC_res,TPC_res_err,EPD_res,EPD_res_err" << endl;
    for (int i=0; i<num_cen; i++)
    {
        resFile << res[0][i] << ",";
        resFile << res_err[0][i] << ",";
        resFile << res[1][i] << ",";
        resFile << res_err[1][i] << endl;
    }
    resFile.close();

    // generate TOF efficiency
    gSystem->Exec(Form("root -l -b -q 'draw_TOF_eff_2.cpp(\"%s\", \"TOFEfficiency.root\", %f)'", inFileName, y_cut));
    TFile *feff = new TFile("./TOFEfficiency.root", "READ");

    //const char* particle = "piminus"; // "pipluslus", "piminus", "kplus", "kminus", "proton", "antiproton"
    //const char* EP_method = "TPC"; // "TPC", "EPD"
    const char* particles[num_par] = {"piplus", "piminus", "proton", "antiproton", "kplus", "kminus"};
    const char* EP_methods[num_EP] = {"TPC", "EPD"};

    float v2[num_par][num_EP][num_cen] = {0};
    float v2_err[num_par][num_EP][num_cen] = {0};
    float v2_counts[num_par][num_cen] = {0};

    float v2_pt[num_par][num_EP][num_cen][num_pt_bins] = {0};
    float v2_pt_err[num_par][num_EP][num_cen][num_pt_bins] = {0};
    float v2_pt_counts[num_par][num_cen][num_pt_bins] = {0};

    // systematic uncertainty
    if (sys_tag == 5) pT_lo_nq = 0.3;

    for (int par=0; par<num_par; par++)
    {
        const char* particle = particles[par];
        for (int ep=0; ep<num_EP; ep++)
        {
            const char* EP_method = EP_methods[ep];
            float pT_min = 0.2;
            float pT_max = 2.0;
            float pT_TOFth = 0.2;
            if (!strcmp(particle, "piplus") || !strcmp(particle, "piminus")) {pT_TOFth = 0.4; pT_min = pT_lo_nq*2; pT_max = pT_hi_nq*2;} // 0.105 - 1.86 mT range for 0.2 - 2.0 pT
            if (!strcmp(particle, "kplus") || !strcmp(particle, "kminus")) {pT_TOFth = 0.4; pT_min = pT_lo_nq*2; pT_max = pT_hi_nq*2;} // 0.25 - 2.0 mT range for 0.2 - 2.0 pT
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
                if (!strcmp(particle, "kplus") || !strcmp(particle, "kminus"))
                {
                    pT_min = sqrt((pT_min+kion_mass) * (pT_min+kion_mass) - kion_mass * kion_mass);
                    pT_max = sqrt((pT_max+kion_mass) * (pT_max+kion_mass) - kion_mass * kion_mass);
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
            for (int cen=1; cen<=num_cen; cen++)
            {
                TProfile *p = nullptr;
                if (!strcmp(EPD_method, "1st")) p = (TProfile*)f->Get(Form("h%s_%s_v2_pt_%d_1st", particle, EP_method, cen));
                else if (!strcmp(EPD_method, "2nd")) p = (TProfile*)f->Get(Form("h%s_%s_v2_pt_%d_2nd", particle, EP_method, cen));
                else throw invalid_argument("Invalid EPD method!");
                if (!p) p = (TProfile*)f->Get(Form("h%s_%s_v2_pt_%d", particle, EP_method, cen));
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
                Double_t cuts[num_par] = {cut1, cut2, cut3, cut4};
                TProfile* hv2_pt_new = (TProfile*)hv2_pt->Rebin(3, "hv2_pt_new", cuts);
                hv2_pt_new->SetErrorOption("");
                float v2_raw = hv2_pt_new->GetBinContent(2);
                float v2_raw_err = hv2_pt_new->GetBinError(2);
                // float res_val = res[ep][cen-1];
                // float res_val_err = res_err[ep][cen-1];
                // v2[par][ep][cen-1] = v2_raw / res_val;
                // v2_err[par][ep][cen-1] = sqrt(pow(v2_raw_err / v2_raw, 2)+ pow(res_val_err / res_val, 2)) * fabs(v2[par][ep][cen-1]);
                v2[par][ep][cen-1] = v2_raw;
                v2_err[par][ep][cen-1] = v2_raw_err;
                v2_counts[par][cen-1] = hv2_pt_new->GetBinEffectiveEntries(2);

                // v2 vs pT
                hv2_pt->SetErrorOption("");
                for (int i = 0; i < hv2_pt->GetNbinsX(); i++)
                {
                    float pt = hv2_pt->GetBinCenter(i + 1);
                    if (pt < 0.0 || pt > 2.0) continue;
                    if (i >= num_pt_bins) continue;
                    float v2_raw = hv2_pt->GetBinContent(i + 1);
                    float v2_raw_err = hv2_pt->GetBinError(i + 1);
                    v2_pt[par][ep][cen-1][i] = v2_raw;
                    v2_pt_err[par][ep][cen-1][i] = v2_raw_err;
                    v2_pt_counts[par][cen-1][i] = hv2_pt->GetBinEntries(i + 1);
                }
            }
        }
    }
    // pretty print v2 and v2 err to csv
    ofstream outputFile(outFileName);
    outputFile << "piplus_counts,piminus_counts,proton_counts,antiproton_counts,kplus_counts,kminus_counts,";
    outputFile << "piplus_v2_TPC,piminus_v2_TPC,proton_v2_TPC,antiproton_v2_TPC,kplus_v2_TPC,kminus_v2_TPC,";
    outputFile << "piplus_v2_err_TPC,piminus_v2_err_TPC,proton_v2_err_TPC,antiproton_v2_err_TPC,kplus_v2_err_TPC,kminus_v2_err_TPC,";
    outputFile << "piplus_v2_EPD,piminus_v2_EPD,proton_v2_EPD,antiproton_v2_EPD,kplus_v2_EPD,kminus_v2_EPD,";
    outputFile << "piplus_v2_err_EPD,piminus_v2_err_EPD,proton_v2_err_EPD,antiproton_v2_err_EPD,kplus_v2_err_EPD,kminus_v2_err_EPD" << endl;
    for (int i=0; i<num_cen; i++)
    {
        for (int j=0; j<num_par; j++) outputFile << v2_counts[j][i] << ",";
        for (int j=0; j<num_par; j++) outputFile << v2[j][0][i] << ",";
        for (int j=0; j<num_par; j++) outputFile << v2_err[j][0][i] << ",";
        for (int j=0; j<num_par; j++) outputFile << v2[j][1][i] << ",";
        for (int j=0; j<num_par-1; j++) outputFile << v2_err[j][1][i] << ",";
        outputFile << v2_err[num_par-1][1][i] << endl;
    }
    outputFile.close();

    // v2 vs pT
    for (int cen=0; cen<num_cen; cen++)
    {
        std::string outFileNameStr_cen(outFileName);
        outFileNameStr_cen.replace(outFileNameStr_cen.find(".csv"), 4, Form("_cen%d.csv", cen+1));
        ofstream outputFile_cen(outFileNameStr_cen);
        outputFile_cen << "piplus_counts,piminus_counts,proton_counts,antiproton_counts,kplus_counts,kminus_counts,";
        outputFile_cen << "piplus_v2_TPC,piminus_v2_TPC,proton_v2_TPC,antiproton_v2_TPC,kplus_v2_TPC,kminus_v2_TPC,";
        outputFile_cen << "piplus_v2_err_TPC,piminus_v2_err_TPC,proton_v2_err_TPC,antiproton_v2_err_TPC,kplus_v2_err_TPC,kminus_v2_err_TPC,";
        outputFile_cen << "piplus_v2_EPD,piminus_v2_EPD,proton_v2_EPD,antiproton_v2_EPD,kplus_v2_EPD,kminus_v2_EPD,";
        outputFile_cen << "piplus_v2_err_EPD,piminus_v2_err_EPD,proton_v2_err_EPD,antiproton_v2_err_EPD,kplus_v2_err_EPD,kminus_v2_err_EPD" << endl;
        for (int ptbin=0; ptbin<num_pt_bins; ptbin++)
        {
            for (int j=0; j<num_par; j++) outputFile_cen << v2_pt_counts[j][cen][ptbin] << ",";
            for (int j=0; j<num_par; j++) outputFile_cen << v2_pt[j][0][cen][ptbin] << ",";
            for (int j=0; j<num_par; j++) outputFile_cen << v2_pt_err[j][0][cen][ptbin] << ",";
            for (int j=0; j<num_par; j++) outputFile_cen << v2_pt[j][1][cen][ptbin] << ",";
            for (int j=0; j<num_par-1; j++) outputFile_cen << v2_pt_err[j][1][cen][ptbin] << ",";
            outputFile_cen << v2_pt_err[num_par-1][1][cen][ptbin] << endl;
        }
        outputFile_cen.close();
    }
    f->Close();
    feff->Close();
}
