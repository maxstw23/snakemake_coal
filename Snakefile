configfile: 'config.yaml'

import numpy as np
import glob
import re
import uproot

# to be changed
energies = config['energies']
# grab files of the form result*_{energy}.root
data_files = {'0': {energy: sorted(glob.glob(f'data/result*_{energy}.root'), key=lambda x: int(re.search(r'\d+', x).group()))[-1] for energy in energies}}
data_files.update({str(sys_tag): {energy: sorted(glob.glob(f'data/sys_tag_{sys_tag}/result*_{energy}.root'), key=lambda x: int(re.search(r'\d+', x).group()))[-1] for energy in energies} for sys_tag in [1,2]})

# check whether the data files contain 2D histograms
use_2D = {energy: 1 if 'hpiplus_EPD_v2_y_pt_1;1' in uproot.open(data_files['0'][energy]).keys() else 0 for energy in energies}

rule all:
    input: #'plots/coal_report.pdf',
           'plots/final/report.pdf',
           expand('plots/sys_tag_0/energy_{energy}/delta_pion.pdf', energy=energies),
           expand('plots/sys_tag_0/energy_{energy}/coal_combined.pdf', energy=energies),
           expand('result/sys_tag_0/energy_{energy}/v2_noeff_corrected.csv', energy=energies)

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
                eff_lib='scripts/{energy}/Efficiency_cpp.so',
                eff_header='scripts/{energy}/Efficiency.h',
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
                target_dir=`dirname {input.eff_header}`
                # target_dir=temp/sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}_eff
                # rm -rf $target_dir
                # mkdir -p $target_dir
                cp {input.script} $target_dir
                cp {input.tof_script} $target_dir
                cp {input.preprocess_script} $target_dir
                cp scripts/ExtendedTProfile* $target_dir
                cd $target_dir
                root -b -q -l {params.script_name}\(\\"$cwd/{input.data_file}\\"\,\\"$cwd/{output.v2}\\"\,{params.eta_cut}\,{params.pt_lo}\,{params.pt_hi}\,\\"{params.EPD_method}\\"\,{params.use_2D}\,{params.use_mT}\) > $cwd/{log.stdout} 2> $cwd/{log.stderr}
                rm -f TOFEfficiency.root
                rm -f preprocessed*
                cd ..
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
                cwd=$(pwd)
                target_dir=temp/sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}
                rm -rf $target_dir
                mkdir -p $target_dir
                cp {input.script} $target_dir
                cp {input.tof_script} $target_dir
                cp {input.preprocess_script} $target_dir
                cp scripts/ExtendedTProfile* $target_dir
                cd $target_dir
                root -b -q -l {params.script_name}\(\\"$cwd/{input.data_file}\\"\,\\"$cwd/{output.v2}\\"\,{params.eta_cut}\,{params.pt_lo}\,{params.pt_hi}\,\\"{params.EPD_method}\\"\,{params.use_2D}\,{params.use_mT}\) > $cwd/{log.stdout} 2> $cwd/{log.stderr}
                rm -f TOFEfficiency.root
                rm -f preprocessed*
                cd $cwd
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

rule plot_v2:
    input: data_points='result/sys_tag_{sys_tag}/energy_{energy}/v2_noeff_corrected.csv',
           data_points_lambda='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_EPD.csv',
           data_points_lambdabar='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_EPD.csv',
           data_points_lambda_TPC='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_TPC.csv',
           data_points_lambdabar_TPC='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_TPC.csv',
           script='scripts/plot_v2_new.py',
           resolution='result/sys_tag_0/energy_{energy}/v2_noeff_corrected_res.csv'
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
            'plots/sys_tag_{sys_tag}/energy_{energy}/lambda_delta_v2_EPD.yaml'
    params: yrange_lo=lambda wildcards: config['plotting']['yrange_lo'][wildcards.energy] if config['plotting']['yrange_strategy'] == 'manual' else None,
            yrange_hi=lambda wildcards: config['plotting']['yrange_hi'][wildcards.energy] if config['plotting']['yrange_strategy'] == 'manual' else None
    log: stdout='logs/sys_tag_{sys_tag}/energy_{energy}/plot_v2_eff_correction.log', stderr='logs/sys_tag_{sys_tag}/energy_{energy}/plot_v2_eff_correction.err'
    shell:
        """
        python {input.script} '{input.data_points}' '{input.data_points_lambda}' '{input.data_points_lambdabar}' '{input.resolution}' 'plots/sys_tag_{wildcards.sys_tag}/energy_{wildcards.energy}/' '{wildcards.energy}' {params.yrange_lo} {params.yrange_hi} > {log.stdout} 2> {log.stderr}       
        """

rule plot_v2_special_sys:
    input: data_points='result/special_sys_tag_{sys_tag}/energy_{energy}/v2_noeff_corrected.csv',
           data_points_lambda='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_EPD.csv',
           data_points_lambdabar='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_EPD.csv',
           data_points_lambda_TPC='result/sys_tag_0/energy_{energy}/fit_Lambda_v2_TPC.csv',
           data_points_lambdabar_TPC='result/sys_tag_0/energy_{energy}/fit_Lambdabar_v2_TPC.csv',
           script='scripts/plot_v2_new.py',
           resolution='result/sys_tag_0/energy_{energy}/v2_noeff_corrected_res.csv'
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
    input: data_points=expand('result/sys_tag_0/energy_{energy}/v2_noeff_corrected.csv', energy=energies),
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
            sys_tag=[1,2], energy=wildcards.energy, EP_method=wildcards.EP_method
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

rule generate_paper_plots:
    input: script='scripts/generate_paper_plots.py',
           res=expand('result/sys_tag_0/energy_{energy}/v2_noeff_corrected_res.csv', energy=energies), # ['7p7GeV', '14p6GeV', '19p6GeV', '27GeV']),
           ratio=expand('plots/final/energy_{energy}/coal_{EP_method}.yaml', energy=energies, EP_method=['TPC', 'EPD']),
           delta_v2=expand('plots/sys_tag_0/energy_{energy}/lambda_delta_v2_{EP_method}.yaml', energy=energies, EP_method=['TPC', 'EPD']),
           delta_v2_isobar=expand('plots/sys_tag_0/energy_{energy}/pi_delta_v2_{EP_method}.yaml', energy=['isobar_Ru', 'isobar_Zr'], EP_method=['TPC', 'EPD']),
           # v2=expand('result/final/energy_{energy}/v2_noeff_corrected.csv', energy=energies),
           # v2=expand('result/sys_tag_0/energy_{energy}/v2_noeff_corrected.csv', energy=energies),
           v2=expand('result/sys_tag_0/energy_{energy}/v2_noeff_corrected.csv', energy=energies),
           v2_eff=expand('result/sys_tag_0/energy_{energy}/v2_eff_corrected.csv', energy=['7p7GeV', '14p6GeV', '19p6GeV', '27GeV'])
    output: 'plots/final/report.pdf'
    log: stdout='logs/generate_paper_plots.log', stderr='logs/generate_paper_plots.err'
    shell:
        'python {input.script} --input_res {input.res} --input_ratio {input.ratio} --input_delta_v2 {input.delta_v2} --input_delta_v2_isobar {input.delta_v2_isobar} --input_v2 {input.v2} --input_v2_eff {input.v2_eff} --output {output}'