configfile: 'config.yaml'

import numpy as np
import glob
import re
import uproot

# CERN ROOT via Docker (not installed locally)
DOCKER_ROOT = 'docker run --rm -v "$(pwd)":/work -w /work rootproject/root:latest root'

# to be changed
energies = config['energies']
# grab files of the form result*_{energy}.root
data_files = {'0': {energy: sorted(glob.glob(f'data/result*_{energy}.root'), key=lambda x: int(re.search(r'\d+', x).group()))[-1] for energy in energies}}

# Systematic-variation tags (1: tighter Vz<35; 2: nHitsFit>=20; 4: looser primary-track PID nSigma 2->3).
# A tag may cover only a subset of energies, so register only the (tag, energy)
# pairs whose ROOT file actually exists -- a missing energy must not break DAG construction.
SYS_TAGS = [1, 2, 4]
for _sys_tag in SYS_TAGS:
    _matches = {}
    for energy in energies:
        _hits = sorted(glob.glob(f'data/sys_tag_{_sys_tag}/result*_{energy}.root'),
                       key=lambda x: int(re.search(r'\d+', x).group()))
        if _hits:
            _matches[energy] = _hits[-1]
    data_files[str(_sys_tag)] = _matches

# Systematic tags that actually have data for a given energy (sizes the sys band per energy).
def sys_tags_for(energy):
    return [t for t in SYS_TAGS if energy in data_files[str(t)]]

# check whether the data files contain 2D histograms
use_2D = {energy: 1 if 'hpiplus_EPD_v2_y_pt_1;1' in uproot.open(data_files['0'][energy]).keys() else 0 for energy in energies}

# Pick eff- or noeff-corrected v2 CSV based on config['correct_eff'].
# suffix = '' / '_res' / '_cen{N}' to address the integrated, resolution, or per-pT files.
def v2_csv(sys_tag, energy, suffix=''):
    tag = 'eff' if config.get('correct_eff', 0) else 'noeff'
    return f'result/sys_tag_{sys_tag}/energy_{energy}/v2_{tag}_corrected{suffix}.csv'

# ---------------------------------------------------------------------------
# Alternative event-plane "planes" (parallel trees built in the SAME run)
# ---------------------------------------------------------------------------
# The default leg (2nd-order participant) is untouched: it keeps writing to the
# bare result/ + plots/ trees using config['EPD_method'] (= "2nd").
# Each entry below adds a *parallel* tree result/<plane>/... + plots/<plane>/...
# that re-runs the pion-coalescence chain with a different EPD harmonic, selected
# by the value (the EPD_method string passed to the v2 scripts; "1st" -> _1st
# histograms + hEPDEP_ew_cos_1 resolution, "2nd" -> _2nd + hEPDEP_ew_cos_2).
#   spectator_1st -> "1st"  (1st-order spectator plane)
# NOTE: only the *pion* coalescence panels switch plane. The Lambda Delta-v2 and
# isobar panels in the spectator report deliberately reuse the default 2nd-order
# outputs (Lambda/NCQ are out of scope here); see generate_paper_plots_plane.
#
# Each plane maps to (EPD_method, source):
#   EPD_method -- "1st"/"2nd" harmonic passed to the v2 scripts (selects the
#                 _1st/_2nd histograms + hEPDEP_ew_cos_1/2 resolution);
#   source     -- which ROOT production to read:
#                 'default'    -> the standard data_files[sys_tag][energy] file,
#                                 in which "1st"=spectator and "2nd"=participant;
#                 'part_tag_1' -> the swapped-plane production (data/part_tag_1/),
#                                 in which "1st"=PARTICIPANT and "2nd"=SPECTATOR.
# Together the default tree (2nd-order participant), spectator_1st (1st-order
# spectator) and the two part_tag_1 cross-planes (1st participant, 2nd spectator)
# span the full {1st,2nd} x {participant,spectator} matrix used to disentangle
# harmonic order from plane type (see scripts/plane_matrix.py).
ALT_PLANES = {
    'spectator_1st':   ('1st', 'default'),
    'participant_1st': ('1st', 'part_tag_1'),
    'spectator_2nd':   ('2nd', 'part_tag_1'),
}

# part_tag_1 swapped-plane production files (only the energies that actually exist).
part_tag_files = {}
for energy in energies:
    _hits = sorted(glob.glob(f'data/part_tag_1/result*_{energy}.root'),
                   key=lambda x: int(re.search(r'\d+', x).group()))
    if _hits:
        part_tag_files[energy] = _hits[-1]

# Resolve the raw ROOT file feeding an alt-plane (sys_tag is ignored for part_tag_1,
# which is a single dedicated production).
def plane_data_file(plane, sys_tag, energy):
    src = ALT_PLANES[plane][1]
    if src == 'default':
        return data_files[sys_tag][energy]
    return part_tag_files[energy]

# Energies available for a given plane (part_tag_1 currently ships 14p6GeV only).
def energies_for_plane(plane):
    if ALT_PLANES[plane][1] == 'default':
        return list(energies)
    return [e for e in energies if e in part_tag_files]

# Planes that get the full multi-energy paper report. The part_tag_1 cross-planes
# ship a single energy, so they only produce per-energy v2 CSVs consumed by the
# plane-matrix comparison -- they are not run through generate_paper_plots_plane.
REPORT_PLANES = [p for p in ALT_PLANES if ALT_PLANES[p][1] == 'default']

def v2_csv_plane(plane, sys_tag, energy, suffix=''):
    tag = 'eff' if config.get('correct_eff', 0) else 'noeff'
    return f'result/{plane}/sys_tag_{sys_tag}/energy_{energy}/v2_{tag}_corrected{suffix}.csv'

rule all:
    input: #'plots/coal_report.pdf',
           'plots/final/report.pdf',
           expand('plots/sys_tag_0/energy_{energy}/delta_pion.pdf', energy=energies),
           expand('plots/sys_tag_0/energy_{energy}/coal_combined.pdf', energy=energies),
           [v2_csv('0', e) for e in energies],
           # alternative-plane (e.g. 1st-order spectator) final plots, built in the same run
           expand('plots/{plane}/final/report.pdf', plane=REPORT_PLANES),
           expand('plots/{plane}/sys_tag_0/energy_{energy}/delta_pion.pdf', plane=REPORT_PLANES, energy=energies),
           expand('plots/{plane}/sys_tag_0/energy_{energy}/coal_combined.pdf', plane=REPORT_PLANES, energy=energies),
           # 2x2 plane-matrix comparison (order x plane-type) for each energy that has a part_tag_1 production
           expand('plots/plane_matrix/energy_{energy}/plane_matrix.pdf', energy=sorted(part_tag_files))

rule generate_report:
    input: expand('plots/sys_tag_0/energy_{energy}/coal_combined.pdf', energy=energies),
           'plots/coal_all.pdf',
           'plots/coal_peri.pdf'
    output: 'plots/coal_report.pdf'
    shell:
        'python -m fitz join -o {output} {input}'

rule TPC_eff:
    input: paths=lambda wildcards: expand('data/embedding/{energy}/cen{cent}.{particle}.root', cent=np.arange(1,10), \
           particle=['piplus', 'piminus', 'proton', 'antiproton', 'kplus', 'kminus'], energy=wildcards.energy),
           script='scripts/TPC_eff.py',
           header='scripts/Efficiency.h'
    output: source='scripts/{energy}/Efficiency.cpp',
            header='scripts/{energy}/Efficiency.h'
    params: pt_lo=lambda wildcards: config['pt_fit_lo'][wildcards.energy],
            pt_hi=lambda wildcards: config['pt_fit_hi'][wildcards.energy],
            eta_cut=config['eta_cut']
    log: stdout='logs/{energy}/TPC_eff.log', stderr='logs/{energy}/TPC_eff.err'
    shell:
        """
        cp {input.header} {output.header}
        mkdir -p scripts/{wildcards.energy}
        mkdir -p plots/QA/
        python {input.script} {wildcards.energy} {params.eta_cut} {input.paths} {params.pt_lo} {params.pt_hi} > {log.stdout} 2> {log.stderr}
        """

rule compile_efficiency:
    input: 'scripts/{energy}/Efficiency.cpp',
           'scripts/{energy}/Efficiency.h'
    output: 'scripts/{energy}/Efficiency_cpp.so',
            temp('scripts/{energy}/compile_efficiency.cpp')
    log: stdout='logs/{energy}/compile_efficiency.log', stderr='logs/{energy}/compile_efficiency.err'
    shell:
        """
        cwd=$(pwd)
        cd scripts/{wildcards.energy}
        touch compile_efficiency.cpp
        echo "void compile_efficiency() {{" > compile_efficiency.cpp
        echo 'gROOT->ProcessLine(".L Efficiency.cpp+");}}' >> compile_efficiency.cpp
        root -b -q -l compile_efficiency.cpp > $cwd/{log.stdout} 2> $cwd/{log.stderr}
        """

rule compile_user_class:
    input: 'scripts/ExtendedTProfile.cpp',
           'scripts/ExtendedTProfile.h'
    output: 'scripts/ExtendedTProfile_cpp.so',
            temp('scripts/compile_user_class.cpp')
    log: stdout='logs/compile_user_class.log', stderr='logs/compile_user_class.err'
    shell:
        """
        cwd=$(pwd)
        cd scripts
        touch compile_user_class.cpp
        echo "void compile_user_class() {{" > compile_user_class.cpp
        echo 'gROOT->ProcessLine(".L ExtendedTProfile.cpp+");}}' >> compile_user_class.cpp
        root -b -q -l compile_user_class.cpp > $cwd/{log.stdout} 2> $cwd/{log.stderr}
        """

rule v2_eff_correction:
        input: user_class_lib='scripts/ExtendedTProfile_cpp.so',
                user_class_header='scripts/ExtendedTProfile.h',
                eff_header='scripts/{energy}/Efficiency.h',
                eff_cpp='scripts/{energy}/Efficiency.cpp',
                tof_script = 'scripts/draw_TOF_eff_2.cpp',
                preprocess_script = 'scripts/coal_preprocess.cpp',
                script = 'scripts/v2_eff_correction.cpp',
                data_file = lambda wildcards: data_files[wildcards.sys_tag][wildcards.energy]
        output: v2='result/sys_tag_{sys_tag}/energy_{energy}/v2_eff_corrected.csv',
                res='result/sys_tag_{sys_tag}/energy_{energy}/v2_eff_corrected_res.csv'
        params: pt_lo=lambda wildcards: config['ptnq_lo'][wildcards.energy],
                pt_hi=lambda wildcards: config['ptnq_hi'][wildcards.energy],
                eta_cut=config['eta_cut'],
                script_name = 'v2_eff_correction.cpp',
                use_2D=lambda wildcards: use_2D[wildcards.energy],
                use_mT=config['use_mT'],
                EPD_method=lambda wildcards: config['EPD_method'][wildcards.energy]
        log: stdout='logs/sys_tag_{sys_tag}/energy_{energy}/v2_eff_correction.log', stderr='logs/sys_tag_{sys_tag}/energy_{energy}/v2_eff_correction.err'
        shell:
                """
                cwd=$(pwd)
                target_dir=temp/sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}
                rm -rf $target_dir
                mkdir -p $target_dir
                cp {input.script} $target_dir
                cp {input.tof_script} $target_dir
                cp {input.preprocess_script} $target_dir
                cp scripts/ExtendedTProfile.h $target_dir
                cp scripts/ExtendedTProfile.cpp $target_dir
                cp {input.eff_header} $target_dir
                cp {input.eff_cpp} $target_dir
                docker run --rm -v "$(pwd)":/work -w /work/$target_dir rootproject/root:latest \
                    bash -c 'root -b -q -l -e ".L ExtendedTProfile.cpp+" -e ".L Efficiency.cpp+" && root -b -q -l '"'"'{params.script_name}("/work/{input.data_file}","/work/{output.v2}",{params.eta_cut},{params.pt_lo},{params.pt_hi},"{params.EPD_method}",{params.use_2D},{params.use_mT},{wildcards.sys_tag})'"'"'' > {log.stdout} 2> {log.stderr}
                rm -f $target_dir/TOFEfficiency.root
                rm -f $target_dir/preprocessed*
                """

rule v2_no_eff_correction:
        input: user_class_lib='scripts/ExtendedTProfile_cpp.so',
                user_class_header='scripts/ExtendedTProfile.h',
                tof_script = 'scripts/draw_TOF_eff_2.cpp',
                preprocess_script = 'scripts/coal_preprocess.cpp',
                script = 'scripts/v2_no_eff_correction.cpp',
                # script = 'scripts/v2_no_eff_correction_alt.cpp',
                data_file = lambda wildcards: data_files[wildcards.sys_tag][wildcards.energy]
        output: v2='result/sys_tag_{sys_tag}/energy_{energy}/v2_noeff_corrected.csv',
                res='result/sys_tag_{sys_tag}/energy_{energy}/v2_noeff_corrected_res.csv'
        params: pt_lo=lambda wildcards: config['ptnq_lo'][wildcards.energy],
                pt_hi=lambda wildcards: config['ptnq_hi'][wildcards.energy],
                eta_cut=config['eta_cut'],
                script_name = 'v2_no_eff_correction.cpp',
                # script_name = 'v2_no_eff_correction_alt.cpp',
                use_2D=lambda wildcards: use_2D[wildcards.energy],
                use_mT=config['use_mT'],
                EPD_method=lambda wildcards: config['EPD_method'][wildcards.energy],
                energy=lambda wildcards: wildcards.energy
        log: stdout='logs/sys_tag_{sys_tag}/energy_{energy}/v2_no_eff_correction.log', stderr='logs/sys_tag_{sys_tag}/energy_{energy}/v2_no_eff_correction.err'
        shell:
                """
                target_dir=temp/sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}
                rm -rf $target_dir
                mkdir -p $target_dir
                cp {input.script} $target_dir
                cp {input.tof_script} $target_dir
                cp {input.preprocess_script} $target_dir
                cp scripts/ExtendedTProfile.h $target_dir
                cp scripts/ExtendedTProfile.cpp $target_dir
                docker run --rm -v "$(pwd)":/work -w /work/$target_dir rootproject/root:latest \
                    bash -c 'root -b -q -l -e ".L ExtendedTProfile.cpp+" && root -b -q -l '"'"'{params.script_name}("/work/{input.data_file}","/work/{output.v2}",{params.eta_cut},{params.pt_lo},{params.pt_hi},"{params.EPD_method}",{params.use_2D},{params.use_mT})'"'"'' > {log.stdout} 2> {log.stderr}
                rm -f $target_dir/TOFEfficiency.root
                rm -f $target_dir/preprocessed*
                """

rule v2_no_eff_correction_special_sys:
        input: user_class_lib='scripts/ExtendedTProfile_cpp.so',
                user_class_header='scripts/ExtendedTProfile.h',
                tof_script = 'scripts/draw_TOF_eff_2.cpp',
                preprocess_script = 'scripts/coal_preprocess.cpp',
                script = 'scripts/v2_no_eff_correction.cpp',
                # script = 'scripts/v2_no_eff_correction_alt.cpp',
                data_file = lambda wildcards: data_files['0'][wildcards.energy]
        output: v2='result/special_sys_tag_{sys_tag}/energy_{energy}/v2_noeff_corrected.csv',
                res='result/special_sys_tag_{sys_tag}/energy_{energy}/v2_noeff_corrected_res.csv'
        params: pt_lo=lambda wildcards: config['ptnq_lo'][wildcards.energy],
                pt_hi=lambda wildcards: config['ptnq_hi'][wildcards.energy],
                eta_cut=config['eta_cut'],
                script_name = 'v2_no_eff_correction.cpp',
                # script_name = 'v2_no_eff_correction_alt.cpp',
                use_2D=lambda wildcards: use_2D[wildcards.energy],
                use_mT=config['use_mT'],
                EPD_method=lambda wildcards: config['EPD_method'][wildcards.energy],
                energy=lambda wildcards: wildcards.energy
        log: stdout='logs/special_sys_tag_{sys_tag}/energy_{energy}/v2_no_eff_correction.log', stderr='logs/special_sys_tag_{sys_tag}/energy_{energy}/v2_no_eff_correction.err'
        shell:
                """
                cwd=$(pwd)
                target_dir=temp/special_sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}
                rm -rf $target_dir
                mkdir -p $target_dir
                cp {input.script} $target_dir
                cp {input.tof_script} $target_dir
                cp {input.preprocess_script} $target_dir
                cp scripts/ExtendedTProfile* $target_dir
                cd $target_dir
                root -b -q -l {params.script_name}\(\\"$cwd/{input.data_file}\\"\,\\"$cwd/{output.v2}\\"\,{params.eta_cut}\,{params.pt_lo}\,{params.pt_hi}\,\\"{params.EPD_method}\\"\,{params.use_2D}\,{params.use_mT}\,{wildcards.sys_tag}\) > $cwd/{log.stdout} 2> $cwd/{log.stderr}
                rm -f TOFEfficiency.root
                rm -f preprocessed*
                cd $cwd
                """

rule v2_eff_correction_special_sys:
        input: user_class_lib='scripts/ExtendedTProfile_cpp.so',
                user_class_header='scripts/ExtendedTProfile.h',
                eff_lib='scripts/{energy}/Efficiency_cpp.so',
                eff_header='scripts/{energy}/Efficiency.h',
                tof_script = 'scripts/draw_TOF_eff_2.cpp',
                preprocess_script = 'scripts/coal_preprocess.cpp',
                script = 'scripts/v2_eff_correction.cpp',
                data_file = lambda wildcards: data_files['0'][wildcards.energy]
        output: v2='result/special_sys_tag_{sys_tag}/energy_{energy}/v2_eff_corrected.csv',
                res='result/special_sys_tag_{sys_tag}/energy_{energy}/v2_eff_corrected_res.csv'
        params: pt_lo=lambda wildcards: config['ptnq_lo'][wildcards.energy],
                pt_hi=lambda wildcards: config['ptnq_hi'][wildcards.energy],
                eta_cut=config['eta_cut'],
                script_name = 'v2_eff_correction.cpp',
                use_2D=lambda wildcards: use_2D[wildcards.energy],
                use_mT=config['use_mT'],
                EPD_method=lambda wildcards: config['EPD_method'][wildcards.energy]
        log: stdout='logs/special_sys_tag_{sys_tag}/energy_{energy}/v2_eff_correction.log', stderr='logs/special_sys_tag_{sys_tag}/energy_{energy}/v2_eff_correction.err'
        shell:
                """
                cwd=$(pwd)
                target_dir=temp/special_sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}
                rm -rf $target_dir
                mkdir -p $target_dir
                cp {input.script} $target_dir
                cp {input.tof_script} $target_dir
                cp {input.preprocess_script} $target_dir
                cp scripts/ExtendedTProfile* $target_dir
                cp scripts/{wildcards.energy}/Efficiency* $target_dir
                cd $target_dir
                root -b -q -l {params.script_name}\(\\"$cwd/{input.data_file}\\"\,\\"$cwd/{output.v2}\\"\,{params.eta_cut}\,{params.pt_lo}\,{params.pt_hi}\,\\"{params.EPD_method}\\"\,{params.use_2D}\,{params.use_mT}\,{wildcards.sys_tag}\) > $cwd/{log.stdout} 2> $cwd/{log.stderr}
                rm -f TOFEfficiency.root
                rm -f preprocessed*
                cd $cwd
                """

rule plot_v2:
    input: data_points=lambda wildcards: v2_csv(wildcards.sys_tag, wildcards.energy),
           data_points_lambda='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_EPD.csv',
           data_points_lambdabar='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_EPD.csv',
           data_points_lambda_TPC='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_TPC.csv',
           data_points_lambdabar_TPC='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_TPC.csv',
           script='scripts/plot_v2_new.py',
           resolution=lambda wildcards: v2_csv('0', wildcards.energy, '_res')
    output: 'plots/sys_tag_{sys_tag}/energy_{energy}/coal_TPC.pdf',
            'plots/sys_tag_{sys_tag}/energy_{energy}/coal_EPD.pdf',
            'plots/sys_tag_{sys_tag}/energy_{energy}/coal_combined.pdf',
            'plots/sys_tag_{sys_tag}/energy_{energy}/delta_pion.pdf',
            'plots/sys_tag_{sys_tag}/energy_{energy}/coal_TPC.yaml',
            'plots/sys_tag_{sys_tag}/energy_{energy}/coal_EPD.yaml',
            'plots/sys_tag_{sys_tag}/energy_{energy}/delta_pion.yaml',
            'plots/sys_tag_{sys_tag}/energy_{energy}/coal_lambda_TPC.pdf',
            'plots/sys_tag_{sys_tag}/energy_{energy}/coal_lambda_EPD.pdf',
            'plots/sys_tag_{sys_tag}/energy_{energy}/lambda_delta_v2_TPC.yaml',
            'plots/sys_tag_{sys_tag}/energy_{energy}/lambda_delta_v2_EPD.yaml',
            'plots/sys_tag_{sys_tag}/energy_{energy}/ratio_pt_scan.pdf'
    params: yrange_lo=lambda wildcards: config['plotting']['yrange_lo'][wildcards.energy] if config['plotting']['yrange_strategy'] == 'manual' else None,
            yrange_hi=lambda wildcards: config['plotting']['yrange_hi'][wildcards.energy] if config['plotting']['yrange_strategy'] == 'manual' else None
    log: stdout='logs/sys_tag_{sys_tag}/energy_{energy}/plot_v2_eff_correction.log', stderr='logs/sys_tag_{sys_tag}/energy_{energy}/plot_v2_eff_correction.err'
    shell:
        """
        python {input.script} '{input.data_points}' '{input.data_points_lambda}' '{input.data_points_lambdabar}' '{input.resolution}' 'plots/sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}/' '{wildcards.energy}' {params.yrange_lo} {params.yrange_hi} > {log.stdout} 2> {log.stderr}
        """

rule plot_v2_special_sys:
    input: data_points=lambda wildcards: 'result/special_sys_tag_{sys_tag}/energy_{energy}/v2_'.format(**wildcards) + ('eff' if config.get('correct_eff', 0) else 'noeff') + '_corrected.csv',
           data_points_lambda='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_EPD.csv',
           data_points_lambdabar='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_EPD.csv',
           data_points_lambda_TPC='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_TPC.csv',
           data_points_lambdabar_TPC='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_TPC.csv',
           script='scripts/plot_v2_new.py',
           resolution=lambda wildcards: v2_csv('0', wildcards.energy, '_res')
    output: 'plots/special_sys_tag_{sys_tag}/energy_{energy}/coal_TPC.pdf',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/coal_EPD.pdf',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/coal_combined.pdf',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/delta_pion.pdf',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/coal_TPC.yaml',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/coal_EPD.yaml',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/delta_pion.yaml',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/coal_lambda_TPC.pdf',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/coal_lambda_EPD.pdf',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/lambda_delta_v2_TPC.yaml',
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/lambda_delta_v2_EPD.yaml'
    params: yrange_lo=lambda wildcards: config['plotting']['yrange_lo'][wildcards.energy] if config['plotting']['yrange_strategy'] == 'manual' else None,
            yrange_hi=lambda wildcards: config['plotting']['yrange_hi'][wildcards.energy] if config['plotting']['yrange_strategy'] == 'manual' else None
    log: stdout='logs/special_sys_tag_{sys_tag}/energy_{energy}/plot_v2_eff_correction.log', stderr='logs/special_sys_tag_{sys_tag}/energy_{energy}/plot_v2_eff_correction.err'
    shell:
        """
        python {input.script} '{input.data_points}' '{input.data_points_lambda}' '{input.data_points_lambdabar}' '{input.resolution}' 'plots/special_sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}/' '{wildcards.energy}' {params.yrange_lo} {params.yrange_hi} > {log.stdout} 2> {log.stderr}
        """

rule plot_isobar:
    input: data_points_Ru='result/sys_tag_{sys_tag}/energy_isobar_Ru/v2_eff_corrected.csv',
           data_points_Zr='result/sys_tag_{sys_tag}/energy_isobar_Zr/v2_eff_corrected.csv',
           res_Ru='result/sys_tag_{sys_tag}/energy_isobar_Ru/v2_eff_corrected_res.csv',
           res_Zr='result/sys_tag_{sys_tag}/energy_isobar_Zr/v2_eff_corrected_res.csv',
           script='scripts/plot_v2_isobar.py'
    output: 'plots/sys_tag_{sys_tag}/energy_isobar/eq1.pdf',
            'plots/sys_tag_{sys_tag}/energy_isobar/eq2.pdf'
    log: stdout='logs/sys_tag_{sys_tag}/energy_isobar/plot_isobar.log', stderr='logs/sys_tag_{sys_tag}/energy_isobar/plot_isobar.err'
    shell:
        """
        python {input.script} --inputFile '{input.data_points_Ru}' '{input.data_points_Zr}' --inputRes '{input.res_Ru}' '{input.res_Zr}' --outputDir 'plots/isobar/' > {log.stdout} 2> {log.stderr} 
        """

rule plot_all:
    input: data_points=[v2_csv('0', e) for e in energies],
           script='scripts/plot_all.py'
    output: plot_all='plots/coal_all.pdf',
            plot_peri='plots/coal_peri.pdf'
    log: stdout='logs/plot_all.log', stderr='logs/plot_all.err'
    shell:
        """
        python {input.script} --data_paths {input.data_points} --output_paths {output.plot_all} {output.plot_peri} > {log.stdout} 2> {log.stderr}
        """

rule blank:
    output: 'result/blank/{energy}.txt'
    shell: 'touch {output}'

rule combine_sys:
    input: 
        script='scripts/combine_sys.py',
        default='plots/sys_tag_0/energy_{energy}/coal_{EP_method}.yaml',
        regular_sys=lambda wildcards: expand(
            'plots/sys_tag_{sys_tag}/energy_{energy}/coal_{EP_method}.yaml',
            sys_tag=sys_tags_for(wildcards.energy), energy=wildcards.energy, EP_method=wildcards.EP_method
        ),
        special_sys=lambda wildcards: expand(
            'plots/special_sys_tag_{sys_tag}/energy_{energy}/coal_{EP_method}.yaml', 
            sys_tag=[], energy=wildcards.energy, EP_method=wildcards.EP_method
        )
    output: 
        'plots/final/energy_{energy}/coal_{EP_method}.yaml'
    log: 
        stdout='logs/combine_sys_{energy}_{EP_method}.log', 
        stderr='logs/combine_sys_{energy}_{EP_method}.err'
    params:
        # These lambdas prevent "hanging flags" by only adding the flag if files exist
        reg_args = lambda wildcards, input: f"--regular_sys {input.regular_sys}" if input.regular_sys else "",
        spec_args = lambda wildcards, input: f"--special_sys {input.special_sys}" if input.special_sys else ""
    shell:
        """
        python {input.script} \
            --default {input.default} \
            {params.reg_args} \
            {params.spec_args} \
            --output {output} \
            --energy {wildcards.energy} \
            > {log.stdout} 2> {log.stderr}
        """

rule fit_lambda:
    input: data_file=lambda wildcards: data_files[wildcards.sys_tag][wildcards.energy],
           script='scripts/fit_v2.py',
    output: data_points='result/sys_tag_{sys_tag}/energy_{energy}/fit_{particle}_v2_{EP}.csv',
    params: 
        energy=lambda wildcards: wildcards.energy,
        yrebin = 1, # no need to rebin y for v2
        invmass_plot = 'plots/sys_tag_{sys_tag}/paper_yaml/invmass/{particle}_fit_{EP}_{energy}_invmass_cen4_y0.70.8.yaml'
    log: stdout='logs/sys_tag_{sys_tag}/energy_{energy}/fit_{particle}_v2_{EP}.log', stderr='logs/sys_tag_{sys_tag}/energy_{energy}/fit_{particle}_v2_{EP}.err'
    shell: 
        """
        python {input.script} {input.data_file} {output.data_points} --particle {wildcards.particle} --EP {wildcards.EP} --yrebin {params.yrebin} --max_refit 500 --paper_plot_path {params.invmass_plot} > {log.stdout} 2> {log.stderr}
        """

rule rho_coal_ratio:
    input: script='scripts/rho_coal_ratio.py',
           v2=[v2_csv('0', '19p6GeV', f'_cen{cen}') for cen in range(1, 10)],
           res=v2_csv('0', '19p6GeV', '_res'),
           v2_int=v2_csv('0', '19p6GeV')
    output: 'plots/sys_tag_0/energy_19p6GeV/rho_ratio.pdf',
            'plots/sys_tag_0/energy_19p6GeV/bw_fits.pdf',
            expand('plots/sys_tag_0/energy_19p6GeV/rho_v2_diagnostic_{EP}_{ch}.pdf',
                   EP=['TPC','EPD'], ch=['pip','pim']),
    params: n_bootstrap=100, n_steps=5000
    log: stdout='logs/rho_coal_ratio.log', stderr='logs/rho_coal_ratio.err'
    shell:
        """
        python {input.script} --n_bootstrap {params.n_bootstrap} --n_steps {params.n_steps} > {log.stdout} 2> {log.stderr}
        """

BAYES_ENERGIES = ['7p7GeV', '9p2GeV', '11p5GeV', '14p6GeV', '17p3GeV', '19p6GeV', '27GeV']

rule quark_v2_all:
    input: expand('plots/{energy}/trace.nc', energy=BAYES_ENERGIES)

rule plot_energy_dep_bayes:
    input:
        script='scripts/plot_energy_dep_bayes.py',
        traces=expand('plots/{energy}/trace.nc', energy=BAYES_ENERGIES),
        coal=expand('plots/sys_tag_0/energy_{energy}/coal_TPC.yaml', energy=energies),
    output: 'plots/final/energy_dep_bayes.pdf'
    log: stdout='logs/energy_dep_bayes.log', stderr='logs/energy_dep_bayes.err'
    shell:
        """
        python {input.script} --out {output} > {log.stdout} 2> {log.stderr}
        """

rule quark_v2_bayes:
    input:
        script   = 'scripts/quark_v2_bayes.py',
        v2       = lambda wildcards: [v2_csv('0', wildcards.energy, f'_cen{cen}') for cen in [5, 6, 7]],
        res      = lambda wildcards: v2_csv('0', wildcards.energy, '_res'),
        lambda_v2  = lambda wildcards: f'result/sys_tag_0/energy_{wildcards.energy}/fit_Lambda_v2_EPD.csv',
        lambdabar_v2 = lambda wildcards: f'result/sys_tag_0/energy_{wildcards.energy}/fit_Lambdabar_v2_EPD.csv',
    output:
        'plots/{energy}/quark_v2_functions.pdf',
        'plots/{energy}/quark_v2_comparison.pdf',
        'plots/{energy}/transported_signal.pdf',
        'plots/{energy}/posterior_predictive.pdf',
        'plots/{energy}/trace.nc'
    params:
        ep           = 'EPD',
        out_dir      = 'plots/{energy}',
        data_dir     = 'result/sys_tag_0/energy_{energy}',
        csv_prefix   = lambda wildcards: 'v2_eff_corrected' if config.get('correct_eff', 0) else 'v2_noeff_corrected',
        cen_bins     = '5 6 7',
        meson_pt_lo  = 0.16,
        meson_pt_hi  = 2.00,
        baryon_pt_lo = 0.24,
        baryon_pt_hi = 3.00,
    log:
        stdout = 'logs/quark_v2/{energy}/quark_v2_bayes.log',
        stderr = 'logs/quark_v2/{energy}/quark_v2_bayes.err'
    shell:
        """
        mkdir -p logs/quark_v2/{wildcards.energy}
        python {input.script} \
            --energy       {wildcards.energy} \
            --ep           {params.ep} \
            --data_dir     {params.data_dir} \
            --csv_prefix   {params.csv_prefix} \
            --out_dir      {params.out_dir} \
            --cen_bins     {params.cen_bins} \
            --meson_pt_lo  {params.meson_pt_lo} \
            --meson_pt_hi  {params.meson_pt_hi} \
            --baryon_pt_lo {params.baryon_pt_lo} \
            --baryon_pt_hi {params.baryon_pt_hi} \
            > {log.stdout} 2> {log.stderr}
        """

# Parametric centrality variant of quark_v2_bayes for 0-10% and 40-80%.
# The 10-40% case stays in the original rule (output: plots/{energy}/trace.nc).
# This rule writes to plots/{energy}/cen_{cen_range}/ to avoid colliding.
CEN_BINS_MAP = {
    '010':  '8 9',
    '4080': '1 2 3 4',
}
THERMAL_LABELS_MAP = {
    '010':  '00-05 05-10',
    '4080': '40-60 60-80',
}

rule quark_v2_bayes_cen:
    input:
        script   = 'scripts/quark_v2_bayes.py',
        v2       = lambda wildcards: [
            v2_csv('0', wildcards.energy, f'_cen{cen}')
            for cen in CEN_BINS_MAP[wildcards.cen_range].split()
        ],
        res          = lambda wildcards: v2_csv('0', wildcards.energy, '_res'),
        lambda_v2    = lambda wildcards: f'result/sys_tag_0/energy_{wildcards.energy}/fit_Lambda_v2_EPD.csv',
        lambdabar_v2 = lambda wildcards: f'result/sys_tag_0/energy_{wildcards.energy}/fit_Lambdabar_v2_EPD.csv',
    output:
        'plots/{energy}/cen_{cen_range}/quark_v2_functions.pdf',
        'plots/{energy}/cen_{cen_range}/quark_v2_comparison.pdf',
        'plots/{energy}/cen_{cen_range}/transported_signal.pdf',
        'plots/{energy}/cen_{cen_range}/posterior_predictive.pdf',
        'plots/{energy}/cen_{cen_range}/trace.nc',
    wildcard_constraints:
        cen_range = '|'.join(CEN_BINS_MAP.keys())
    params:
        ep             = 'EPD',
        out_dir        = 'plots/{energy}/cen_{cen_range}',
        data_dir       = 'result/sys_tag_0/energy_{energy}',
        csv_prefix     = lambda wildcards: 'v2_eff_corrected' if config.get('correct_eff', 0) else 'v2_noeff_corrected',
        cen_bins       = lambda wildcards: CEN_BINS_MAP[wildcards.cen_range],
        thermal_labels = lambda wildcards: THERMAL_LABELS_MAP[wildcards.cen_range],
        meson_pt_lo    = 0.16,
        meson_pt_hi    = 2.00,
        baryon_pt_lo   = 0.24,
        baryon_pt_hi   = 3.00,
    log:
        stdout = 'logs/quark_v2/{energy}/quark_v2_bayes_cen{cen_range}.log',
        stderr = 'logs/quark_v2/{energy}/quark_v2_bayes_cen{cen_range}.err',
    shell:
        """
        mkdir -p logs/quark_v2/{wildcards.energy}
        python {input.script} \
            --energy             {wildcards.energy} \
            --ep                 {params.ep} \
            --data_dir           {params.data_dir} \
            --csv_prefix         {params.csv_prefix} \
            --out_dir            {params.out_dir} \
            --cen_bins           {params.cen_bins} \
            --thermal_cen_labels {params.thermal_labels} \
            --meson_pt_lo        {params.meson_pt_lo} \
            --meson_pt_hi        {params.meson_pt_hi} \
            --baryon_pt_lo       {params.baryon_pt_lo} \
            --baryon_pt_hi       {params.baryon_pt_hi} \
            > {log.stdout} 2> {log.stderr}
        """

SPECTATOR_DATA_DIR = "/mnt/d/Research/local/strangeness_local/result/14_star"

rule v2_eff_spectator:
    input:
        user_class_header = 'scripts/ExtendedTProfile.h',
        user_class_cpp    = 'scripts/ExtendedTProfile.cpp',
        eff_header        = 'scripts/14p6GeV/Efficiency.h',
        eff_cpp           = 'scripts/14p6GeV/Efficiency.cpp',
        tof_script        = 'scripts/draw_TOF_eff_2.cpp',
        preprocess_script = 'scripts/coal_preprocess.cpp',
        script            = 'scripts/v2_eff_correction.cpp',
        data_file         = SPECTATOR_DATA_DIR + '/{result_file}.root'
    output:
        v2  = 'result/spectator_check/{result_file}_{EPD_method}/v2_eff_corrected.csv',
        res = 'result/spectator_check/{result_file}_{EPD_method}/v2_eff_corrected_res.csv'
    params:
        pt_lo       = config['ptnq_lo']['14p6GeV'],
        pt_hi       = config['ptnq_hi']['14p6GeV'],
        eta_cut     = config['eta_cut'],
        use_2D      = 0,
        use_mT      = config['use_mT'],
        script_name = 'v2_eff_correction.cpp'
    log:
        stdout = 'logs/spectator_check/{result_file}_{EPD_method}/v2_eff_correction.log',
        stderr = 'logs/spectator_check/{result_file}_{EPD_method}/v2_eff_correction.err'
    shell:
        """
        cwd=$(pwd)
        target_dir=temp/spectator_check/{wildcards.result_file}_{wildcards.EPD_method}
        rm -rf $target_dir
        mkdir -p $target_dir
        cp {input.script} $target_dir
        cp {input.tof_script} $target_dir
        cp {input.preprocess_script} $target_dir
        cp {input.user_class_cpp} $target_dir
        cp {input.user_class_header} $target_dir
        cp {input.eff_header} $target_dir
        cp {input.eff_cpp} $target_dir
        docker run --rm \
            -v "$(pwd)":/work \
            -v "{SPECTATOR_DATA_DIR}":/data \
            -w /work/$target_dir \
            rootproject/root:latest \
            bash -c 'root -b -q -l -e ".L ExtendedTProfile.cpp+" -e ".L Efficiency.cpp+" && root -b -q -l '"'"'{params.script_name}("/data/{wildcards.result_file}.root","/work/{output.v2}",{params.eta_cut},{params.pt_lo},{params.pt_hi},"{wildcards.EPD_method}",{params.use_2D},{params.use_mT},0)'"'"'' > $cwd/{log.stdout} 2> $cwd/{log.stderr}
        rm -f $target_dir/TOFEfficiency.root $target_dir/preprocessed*
        """

rule plot_epd_compare:
    input:
        script = 'scripts/plot_epd_compare.py',
        v2  = expand('result/spectator_check/{tag}/v2_eff_corrected.csv',
                     tag=['result24_1st', 'result24_2nd']),
        res = expand('result/spectator_check/{tag}/v2_eff_corrected_res.csv',
                     tag=['result24_1st', 'result24_2nd'])
    output:
        'plots/spectator_check/compare_epd_planes.pdf'
    log:
        stdout = 'logs/spectator_check/plot_epd_compare.log',
        stderr = 'logs/spectator_check/plot_epd_compare.err'
    shell:
        """
        mkdir -p plots/spectator_check
        python {input.script} \
            --labels "1st spectator" "2nd participant" \
            --v2  {input.v2} \
            --res {input.res} \
            --out {output} > {log.stdout} 2> {log.stderr}
        """

rule generate_paper_plots:
    input: script='scripts/generate_paper_plots.py',
           res=[v2_csv('0', e, '_res') for e in energies],
           ratio=expand('plots/final/energy_{energy}/coal_{EP_method}.yaml', energy=energies, EP_method=['TPC', 'EPD']),
           delta_v2=expand('plots/sys_tag_0/energy_{energy}/lambda_delta_v2_{EP_method}.yaml', energy=energies, EP_method=['TPC', 'EPD']),
           delta_v2_isobar=expand('plots/sys_tag_0/energy_{energy}/pi_delta_v2_{EP_method}.yaml', energy=['isobar_Ru', 'isobar_Zr'], EP_method=['TPC', 'EPD']),
           v2=[v2_csv('0', e) for e in energies],
           # v2_eff always points at the eff-corrected CSVs so the side-by-side comparison plot keeps working
           v2_eff=expand('result/sys_tag_0/energy_{energy}/v2_eff_corrected.csv', energy=energies)
    output: 'plots/final/report.pdf'
    log: stdout='logs/generate_paper_plots.log', stderr='logs/generate_paper_plots.err'
    shell:
        'python {input.script} --input_res {input.res} --input_ratio {input.ratio} --input_delta_v2 {input.delta_v2} --input_delta_v2_isobar {input.delta_v2_isobar} --input_v2 {input.v2} --input_v2_eff {input.v2_eff} --output {output}'

# ===========================================================================
# Alternative-plane chain (e.g. spectator_1st): mirrors the default pion
# coalescence chain into a parallel result/<plane>/ + plots/<plane>/ tree.
# Same C++/Python scripts, only EPD_method and output paths change. The bare
# default tree above is left completely untouched; both build in one run.
# ===========================================================================

rule v2_eff_correction_plane:
        input: user_class_lib='scripts/ExtendedTProfile_cpp.so',
                user_class_header='scripts/ExtendedTProfile.h',
                eff_header='scripts/{energy}/Efficiency.h',
                eff_cpp='scripts/{energy}/Efficiency.cpp',
                tof_script = 'scripts/draw_TOF_eff_2.cpp',
                preprocess_script = 'scripts/coal_preprocess.cpp',
                script = 'scripts/v2_eff_correction.cpp',
                data_file = lambda wildcards: plane_data_file(wildcards.plane, wildcards.sys_tag, wildcards.energy)
        output: v2='result/{plane}/sys_tag_{sys_tag}/energy_{energy}/v2_eff_corrected.csv',
                res='result/{plane}/sys_tag_{sys_tag}/energy_{energy}/v2_eff_corrected_res.csv'
        wildcard_constraints: plane='|'.join(ALT_PLANES), sys_tag=r'\d+'
        params: pt_lo=lambda wildcards: config['ptnq_lo'][wildcards.energy],
                pt_hi=lambda wildcards: config['ptnq_hi'][wildcards.energy],
                eta_cut=config['eta_cut'],
                script_name = 'v2_eff_correction.cpp',
                use_2D=lambda wildcards: use_2D[wildcards.energy],
                use_mT=config['use_mT'],
                EPD_method=lambda wildcards: ALT_PLANES[wildcards.plane][0]
        log: stdout='logs/{plane}/sys_tag_{sys_tag}/energy_{energy}/v2_eff_correction.log', stderr='logs/{plane}/sys_tag_{sys_tag}/energy_{energy}/v2_eff_correction.err'
        shell:
                """
                cwd=$(pwd)
                target_dir=temp/{wildcards.plane}/sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}
                rm -rf $target_dir
                mkdir -p $target_dir
                cp {input.script} $target_dir
                cp {input.tof_script} $target_dir
                cp {input.preprocess_script} $target_dir
                cp scripts/ExtendedTProfile.h $target_dir
                cp scripts/ExtendedTProfile.cpp $target_dir
                cp {input.eff_header} $target_dir
                cp {input.eff_cpp} $target_dir
                docker run --rm -v "$(pwd)":/work -w /work/$target_dir rootproject/root:latest \
                    bash -c 'root -b -q -l -e ".L ExtendedTProfile.cpp+" -e ".L Efficiency.cpp+" && root -b -q -l '"'"'{params.script_name}("/work/{input.data_file}","/work/{output.v2}",{params.eta_cut},{params.pt_lo},{params.pt_hi},"{params.EPD_method}",{params.use_2D},{params.use_mT},{wildcards.sys_tag})'"'"'' > {log.stdout} 2> {log.stderr}
                rm -f $target_dir/TOFEfficiency.root
                rm -f $target_dir/preprocessed*
                """

rule plot_v2_plane:
    input: data_points=lambda wildcards: v2_csv_plane(wildcards.plane, wildcards.sys_tag, wildcards.energy),
           # Lambda fits reuse the default 2nd-order plane (Lambda/NCQ out of scope for the plane switch)
           data_points_lambda='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_EPD.csv',
           data_points_lambdabar='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_EPD.csv',
           data_points_lambda_TPC='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_TPC.csv',
           data_points_lambdabar_TPC='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_TPC.csv',
           script='scripts/plot_v2_new.py',
           resolution=lambda wildcards: v2_csv_plane(wildcards.plane, '0', wildcards.energy, '_res')
    output: 'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/coal_TPC.pdf',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/coal_EPD.pdf',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/coal_combined.pdf',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/delta_pion.pdf',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/coal_TPC.yaml',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/coal_EPD.yaml',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/delta_pion.yaml',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/coal_lambda_TPC.pdf',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/coal_lambda_EPD.pdf',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/lambda_delta_v2_TPC.yaml',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/lambda_delta_v2_EPD.yaml',
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/ratio_pt_scan.pdf'
    wildcard_constraints: plane='|'.join(ALT_PLANES), sys_tag=r'\d+'
    params: yrange_lo=lambda wildcards: config['plotting']['yrange_lo'][wildcards.energy] if config['plotting']['yrange_strategy'] == 'manual' else None,
            yrange_hi=lambda wildcards: config['plotting']['yrange_hi'][wildcards.energy] if config['plotting']['yrange_strategy'] == 'manual' else None
    log: stdout='logs/{plane}/sys_tag_{sys_tag}/energy_{energy}/plot_v2_eff_correction.log', stderr='logs/{plane}/sys_tag_{sys_tag}/energy_{energy}/plot_v2_eff_correction.err'
    shell:
        """
        python {input.script} --unmask_epd '{input.data_points}' '{input.data_points_lambda}' '{input.data_points_lambdabar}' '{input.resolution}' 'plots/{wildcards.plane}/sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}/' '{wildcards.energy}' {params.yrange_lo} {params.yrange_hi} > {log.stdout} 2> {log.stderr}
        """

rule combine_sys_plane:
    input:
        script='scripts/combine_sys.py',
        default='plots/{plane}/sys_tag_0/energy_{energy}/coal_{EP_method}.yaml',
        regular_sys=lambda wildcards: expand(
            'plots/{plane}/sys_tag_{sys_tag}/energy_{energy}/coal_{EP_method}.yaml',
            plane=wildcards.plane, sys_tag=sys_tags_for(wildcards.energy), energy=wildcards.energy, EP_method=wildcards.EP_method
        )
    output:
        'plots/{plane}/final/energy_{energy}/coal_{EP_method}.yaml'
    wildcard_constraints: plane='|'.join(ALT_PLANES)
    log:
        stdout='logs/{plane}/combine_sys_{energy}_{EP_method}.log',
        stderr='logs/{plane}/combine_sys_{energy}_{EP_method}.err'
    params:
        reg_args = lambda wildcards, input: f"--regular_sys {input.regular_sys}" if input.regular_sys else ""
    shell:
        """
        python {input.script} \
            --default {input.default} \
            {params.reg_args} \
            --output {output} \
            --energy {wildcards.energy} \
            > {log.stdout} 2> {log.stderr}
        """

rule generate_paper_plots_plane:
    input: script='scripts/generate_paper_plots.py',
           res=lambda wildcards: [v2_csv_plane(wildcards.plane, '0', e, '_res') for e in energies],
           ratio=lambda wildcards: expand('plots/{plane}/final/energy_{energy}/coal_{EP_method}.yaml', plane=wildcards.plane, energy=energies, EP_method=['TPC', 'EPD']),
           # Lambda Delta-v2 and isobar panels reuse the default 2nd-order plane (out of scope for the plane switch)
           delta_v2=expand('plots/sys_tag_0/energy_{energy}/lambda_delta_v2_{EP_method}.yaml', energy=energies, EP_method=['TPC', 'EPD']),
           delta_v2_isobar=expand('plots/sys_tag_0/energy_{energy}/pi_delta_v2_{EP_method}.yaml', energy=['isobar_Ru', 'isobar_Zr'], EP_method=['TPC', 'EPD']),
           v2=lambda wildcards: [v2_csv_plane(wildcards.plane, '0', e) for e in energies],
           v2_eff=lambda wildcards: expand('result/{plane}/sys_tag_0/energy_{energy}/v2_eff_corrected.csv', plane=wildcards.plane, energy=energies)
    output: 'plots/{plane}/final/report.pdf'
    wildcard_constraints: plane='|'.join(ALT_PLANES)
    log: stdout='logs/{plane}/generate_paper_plots.log', stderr='logs/{plane}/generate_paper_plots.err'
    shell:
        'python {input.script} --input_res {input.res} --input_ratio {input.ratio} --input_delta_v2 {input.delta_v2} --input_delta_v2_isobar {input.delta_v2_isobar} --input_v2 {input.v2} --input_v2_eff {input.v2_eff} --output {output} --unmask_epd > {log.stdout} 2> {log.stderr}'

# ===========================================================================
# 2x2 plane-matrix comparison ({1st,2nd} order x {participant,spectator} plane)
# at a single energy: combines the default tree (2nd participant), spectator_1st
# (1st spectator) and the two part_tag_1 cross-planes (1st participant via the
# participant_1st plane, 2nd spectator via spectator_2nd). Holds order fixed to
# isolate the spectator-vs-participant shift, and vice versa. See plane_matrix.py.
# ===========================================================================
rule plane_matrix:
    input: script='scripts/plane_matrix.py',
           attr='scripts/attribute_plane.py',
           default='result/sys_tag_0/energy_{energy}/v2_eff_corrected.csv',
           spectator_1st='result/spectator_1st/sys_tag_0/energy_{energy}/v2_eff_corrected.csv',
           participant_1st='result/participant_1st/sys_tag_0/energy_{energy}/v2_eff_corrected.csv',
           spectator_2nd='result/spectator_2nd/sys_tag_0/energy_{energy}/v2_eff_corrected.csv'
    output: 'plots/plane_matrix/energy_{energy}/plane_matrix.pdf',
            'plots/plane_matrix/energy_{energy}/plane_matrix_v2.pdf',
            'plots/plane_matrix/energy_{energy}/plane_matrix.yaml'
    log: stdout='logs/plane_matrix/energy_{energy}.log', stderr='logs/plane_matrix/energy_{energy}.err'
    shell:
        'python {input.script} --result-dir result --sys-tag 0 --energy {wildcards.energy} --out plots/plane_matrix/energy_{wildcards.energy} > {log.stdout} 2> {log.stderr}'