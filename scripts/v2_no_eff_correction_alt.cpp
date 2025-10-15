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

#define num_par 6
#define num_EP 2
#define num_cen 9

R__LOAD_LIBRARY(./ExtendedTProfile_cpp.so)

using namespace std;

const float PI = TMath::Pi();
const float pion_mass = 0.13957;
const float proton_mass = 0.93827;
// const int energy = 14;

void v2_no_eff_correction_alt(const char* rawFileName, const char* outFileName, float y_cut, float pT_lo_nq, float pT_hi_nq, const char* EPD_method, bool use_2D, bool use_mT=false)
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
        if (res[0][i] <= 0 || res[1][i] <= 0) throw invalid_argument("Resolution is zero!");
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
    const char* particles[num_par] = {"piplus", "piminus", "proton", "antiproton", "antiproton_pip", "antiproton_pim"};
    const char* EP_methods[num_EP] = {"TPC", "EPD"};

    float v2[num_par][num_EP][num_cen] = {0};
    float v2_err[num_par][num_EP][num_cen] = {0};
    float v2_counts[num_par][num_cen] = {0};

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
            if (!strcmp(particle, "kplus") || !strcmp(particle, "kminus")) pT_TOFth = 0.4;
            if (!strcmp(particle, "proton") || !strcmp(particle, "antiproton")) {pT_TOFth = 0.6; pT_min = pT_lo_nq*3; pT_max = pT_hi_nq*3;} // 0.03 - 1.27 mT range for 0.2 - 2.0 pT
            if (!strcmp(particle, "antiproton_pip") || !strcmp(particle, "antiproton_pim")) {pT_TOFth = 0.6; pT_min = pT_lo_nq*3; pT_max = pT_hi_nq*3;} // 0.03 - 1.27 mT range for 0.2 - 2.0 pT
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
                if (!strcmp(particle, "antiproton_pip") || !strcmp(particle, "antiproton_pim")) 
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
            if (!strcmp(particle, "antiproton_pip")) par_str = "AP";
            if (!strcmp(particle, "antiproton_pim")) par_str = "AP";

            // Efficiency *eff = new Efficiency();
            float test1 = 0, test2 = 0;
            float test1_err = 0, test2_err = 0;
            float testpt = 0.5;
            for (int cen=1; cen<=num_cen; cen++)
            {   
                std::string particle_str;
                if (!strcmp(particle, "antiproton_pip") || !strcmp(particle, "antiproton_pim")) particle_str = "antiproton";
                else particle_str = particle;
                TProfile *p = (TProfile*)f->Get(Form("h%s_%s_v2_pt_%d", particle_str.c_str(), EP_method, cen));
                TProfile *p_aux = nullptr; ExtendedTProfile *hv2_pt_aux = nullptr;
                if (!strcmp(particle, "antiproton_pip")) p_aux = (TProfile*)f->Get(Form("h%s_%s_v2_pt_%d", "piplus", EP_method, cen));
                if (!strcmp(particle, "antiproton_pim")) p_aux = (TProfile*)f->Get(Form("h%s_%s_v2_pt_%d", "piminus", EP_method, cen));
                //TH1D *hpT = (TH1D*)f->Get(Form("hgpT_%d", cen));
                //TH1D *hpT_TOF = (TH1D*)f->Get(Form("hgpT_TOF_%d", cen));
                //TEfficiency *hTOF_Eff = new TEfficiency(*hpT_TOF, *hpT);
                TEfficiency *hTOF_Eff = (TEfficiency*)feff->Get(Form("hTOFEff_%d", cen));
                ExtendedTProfile *hv2_pt = new ExtendedTProfile(*p); 
                if (p_aux != nullptr)
                {
                    hv2_pt_aux = new ExtendedTProfile(*p_aux);
                    hv2_pt_aux->Sumw2();
                    hv2_pt_aux->SetErrorOption("s");
                }
                hv2_pt->Sumw2();
                hv2_pt->SetErrorOption("s");

                // efficiency correction
                for (int i = 0; i < hv2_pt->GetNbinsX(); i++)
                {
                    float pt = hv2_pt->GetBinCenter(i + 1);
                    if (pt < pT_min || pt > pT_max) continue;
                    float TOF_eff = 1.0;
                    if (pt > pT_TOFth) TOF_eff = hTOF_Eff->GetEfficiency(hTOF_Eff->FindFixBin(pt));
                    float TPC_eff = 1; // eff->GetEfficiency1D(pt, cen, par_str); //P0[cen-1]*exp(-pow(P1[cen-1]/pt,P2[cen-1]));
                    float scale = 1.0;
                    if (hv2_pt_aux != nullptr) scale = hv2_pt_aux->GetBinEntries(hv2_pt_aux->FindFixBin(pt*2./3.)) / hv2_pt->GetBinEntries(i + 1); 
                    float weight = 1.0 /TOF_eff / TPC_eff * scale;
                    float content = hv2_pt->GetBinContent(i + 1);
                    float error = hv2_pt->GetBinError(i + 1);
                    float entries = hv2_pt->GetBinEntries(i + 1);
                    float entries_eff = hv2_pt->GetBinEffectiveEntries(i + 1);
                    float sumw2 = hv2_pt->GetBinSumw2()->At(i + 1);

                    float W = entries * weight;           // set W(j)
                    float H = content * entries * weight; // set H(j), which is h(j) * W(j)
                    float sumw2_new = sumw2 * weight * weight;
                    float entries_eff_new = W * W / sumw2_new;
                    float E = (error * error + content * content) * entries * weight; // set E(j), error of the "histogram"

                    hv2_pt->SetBinEntries(i + 1, W); // set W(j)
                    hv2_pt->SetBinContent(i + 1, H); // set H(j), which is h(j) * W(j)
                    float entries_eff_current = hv2_pt->GetBinEffectiveEntries(i + 1);
                    hv2_pt->SetBinError(i + 1, sqrt(E)); // set sqrt(E(j))
                    hv2_pt->SetSumw2(i + 1, sumw2_new);

                    // if (i+1 == hv2_pt->FindFixBin(testpt))
                    // {
                    //     hv2_pt->SetErrorOption("");
                    //     test1 = hv2_pt->GetBinContent(hv2_pt->FindFixBin(testpt));
                    //     test1_err = hv2_pt->GetBinError(hv2_pt->FindFixBin(testpt));
                    //     hv2_pt->SetErrorOption("s");
                    // }

                    // if (hv2_pt_aux != nullptr)
                    // {
                    //     int corresponding_bin = hv2_pt_aux->GetXaxis()->FindBin(pt*2./3.);
                    //     // first correct aux in the same way as before
                    //     float TOF_eff_aux = 1.0;
                    //     if (pt*2./3 > pT_TOFth) TOF_eff_aux = hTOF_Eff->GetEfficiency(hTOF_Eff->FindFixBin(pt*2./3.));
                    //     float TPC_eff_aux = 1; // eff->GetEfficiency1D(pt*2./3., cen, par_str); //P0[cen-1]*exp(-pow(P1[cen-1]/pt*2./3.,P2[cen-1]));
                    //     float weight_aux = 1.0 /TOF_eff_aux / TPC_eff_aux;
                    //     float content_aux = hv2_pt_aux->GetBinContent(corresponding_bin);
                    //     float error_aux = hv2_pt_aux->GetBinError(corresponding_bin);
                    //     float entries_aux = hv2_pt_aux->GetBinEntries(corresponding_bin);
                    //     float sumw2_aux = hv2_pt_aux->GetSumw2()->At(corresponding_bin);

                    //     float W_aux = entries_aux * weight_aux;           // set W(j)
                    //     float H_aux = content_aux * entries_aux * weight_aux; // set H(j), which is h(j) * W(j)
                    //     float sumw2_new_aux = sumw2_aux * weight_aux * weight_aux;
                    //     float entries_eff_new_aux = W_aux * W_aux / sumw2_new_aux;
                    //     float E_aux = (error_aux * error_aux + content_aux * content_aux) * entries_aux * weight_aux; // set E(j), sum of weights squared

                    //     hv2_pt_aux->SetBinEntries(corresponding_bin, W_aux); // set W(j)
                    //     hv2_pt_aux->SetBinContent(corresponding_bin, H_aux); // set H(j), which is h(j) * W(j)
                    //     hv2_pt_aux->SetBinError(corresponding_bin, sqrt(E_aux)); // set sqrt(E(j))
                    //     hv2_pt_aux->SetSumw2(corresponding_bin, sumw2_new_aux);

                    //     // now correct profile to have the same number of entries as aux, but keep the same mean and error of the mean
                    //     float W_new = hv2_pt_aux->GetBinEntries(corresponding_bin);
                    //     float scale = W_new / W;
                    //     float H_new = H * scale;
                    //     float E_new = W_new * (hv2_pt->GetBinError(i + 1) * hv2_pt->GetBinError(i + 1) * scale + hv2_pt->GetBinContent(i + 1) * hv2_pt->GetBinContent(i + 1));
                    //     float sumw2_new_new = sumw2_new * scale * scale;

                    //     hv2_pt->SetBinEntries(i + 1, W_new);
                    //     hv2_pt->SetBinContent(i + 1, H_new);
                    //     hv2_pt->SetBinError(i + 1, sqrt(E_new));
                    //     hv2_pt->SetSumw2(i + 1, sumw2_new_new);
                    // }
                }

                // if (hv2_pt_aux != nullptr)
                // {
                //     hv2_pt->SetErrorOption("");
                //     test2 = hv2_pt->GetBinContent(hv2_pt->FindFixBin(testpt));
                //     test2_err = hv2_pt->GetBinError(hv2_pt->FindFixBin(testpt));
                //     std::cout << "particle: " << particle << ", EP method: " << EP_method << ", centrality: " << cen << std::endl;
                //     std::cout << "test1: " << test1 << ", test1_err: " << test1_err << std::endl;
                //     std::cout << "test2: " << test2 << ", test2_err: " << test2_err << std::endl;
                // }
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
            }  
        }
    }
    // pretty print v2 and v2 err to csv
    ofstream outputFile(outFileName);
    outputFile << "piplus_counts,piminus_counts,proton_counts,antiproton_counts,antiproton_pip_counts,antiproton_pim_counts,";
    outputFile << "piplus_v2_TPC,piminus_v2_TPC,proton_v2_TPC,antiproton_v2_TPC,antiproton_pip_v2_TPC,antiproton_pim_v2_TPC,";
    outputFile << "piplus_v2_err_TPC,piminus_v2_err_TPC,proton_v2_err_TPC,antiproton_v2_err_TPC,antiproton_pip_v2_err_TPC,antiproton_pim_v2_err_TPC,";
    outputFile << "piplus_v2_EPD,piminus_v2_EPD,proton_v2_EPD,antiproton_v2_EPD,antiproton_pip_v2_EPD,antiproton_pim_v2_EPD,";
    outputFile << "piplus_v2_err_EPD,piminus_v2_err_EPD,proton_v2_err_EPD,antiproton_v2_err_EPD,antiproton_pip_v2_err_EPD,antiproton_pim_v2_err_EPD" << endl;
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
    f->Close();
    feff->Close();
}
