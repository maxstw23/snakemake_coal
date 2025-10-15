#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
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
#include "TH2D.h"
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

using namespace std;

void draw_TOF_eff_2(const char* inFileName = "./result/7_star/result9.root", const char* outFileName = "TOFEfficiency_7GeV.root", float y_cut=0.6)  
{   
    TString energy = "19GeV";
    bool write2D = true;
    //TFile* f = new TFile("./result/27_star/result79.root", "READ");
    //TFile *f = new TFile("./result/14_star/result4_2.root", "READ");
    TFile *f = new TFile(inFileName, "READ");
    TFile *fout = new TFile(outFileName, "RECREATE");
    //TFile* f = new TFile("./result/14/cf12_eta_cen9.root", "READ");
    //TFile* f = new TFile("./result/14/cf12_eta_cen9_default.root", "READ");

    string p_or_pt = "pT";
    
    for (int j=1; j<=9; j++)
    {
        // TH1D* hpT = (TH1D*)f->Get(Form("hg%s_%d", p_or_pt.c_str(), j));
        // TH1D* hpT_TOF = (TH1D*)f->Get(Form("hg%s_TOF_%d", p_or_pt.c_str(), j));

        TH2D* hgpTeta = (TH2D*)f->Get(Form("hgpTeta_%d", j));
        TH2D* hgpTeta_TOF = (TH2D*)f->Get(Form("hgpTeta_TOF_%d", j));

        int ybin_lo = hgpTeta->GetYaxis()->FindBin(-y_cut);
        int ybin_hi = hgpTeta->GetYaxis()->FindBin(y_cut);
        TH1D* hpT = hgpTeta->ProjectionX(Form("hpT_%d", j), ybin_lo, ybin_hi);
        TH1D* hpT_TOF = hgpTeta_TOF->ProjectionX(Form("hpT_TOF_%d", j), ybin_lo, ybin_hi);
        TEfficiency *Eff = new TEfficiency(*hpT_TOF, *hpT);
        // TEfficiency *Eff2D = new TEfficiency(*hgpTeta_TOF, *hgpTeta);
        TH2D* Eff2D = (TH2D*)hgpTeta_TOF->Clone();
        Eff2D->Sumw2();
        hgpTeta->Sumw2();
        Eff2D->Divide(hgpTeta);
        //cout << Eff->GetEfficiency(Eff->FindFixBin(0.52)) << endl;
        cout << Eff2D->GetBinContent(Eff2D->FindFixBin(0.52, 0.0)) << endl;
        cout << Eff2D->GetBinError(Eff2D->FindFixBin(0.52, 0.0)) << endl;
        // Eff->Draw();

        fout->cd();
        Eff->SetName(Form("hTOFEff_%d", j));
        Eff2D->SetName(Form("hTOFEff_2D_%d", j));
        Eff->Write();
        if (write2D) Eff2D->Write();
    }
    
    fout->Close();
    f->Close();

    // TLegend* l = new TLegend(0.5, 0.9, 0.3, 0.7);
    // l->AddEntry(hO, "w/ #Omega^{-}");
    // l->AddEntry(hOb, "w/ #bar{#Omega}^{+}");
    // l->Draw();
}


