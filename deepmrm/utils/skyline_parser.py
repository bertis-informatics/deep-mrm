from pathlib import Path
import xml.etree.ElementTree as ET
import os

import pandas as pd
import numpy as np

def parse_skyline_file(sky_path):

    sky_path = Path(sky_path)
    #skyline_name = skyline_file_path.stem
    tree = ET.parse(sky_path)
    sample_df = extract_sample_list(tree)
    peptide_df, transition_df = extract_peak_groups(tree)

    # common_prefix = os.path.commonprefix(sample_df['raw_file'].to_list())
    # common_prefix = common_prefix[:common_prefix.rindex('\\')]
    # sample_df['raw_file'] = sample_df['raw_file'].str[len(common_prefix)+1:]

    return sample_df, peptide_df, transition_df
    


def extract_sample_list(skyline_tree):
    root = skyline_tree.getroot()
    replicates = root.find('settings_summary').find('measured_results').findall('replicate')
    rep_infos = []

    for rep in replicates:
        replicate = rep.get('name')
        sample_files = rep.findall('sample_file')
        
        for sample_file in sample_files:
            sample_file_id = sample_file.get('id')
            sample_name = sample_file.get('sample_name')
            file_path = sample_file.get('file_path')
            file_path = file_path.replace('\r', '\\r')
            #file_path="G:\Ruth\Ovarian_cancer_RAW\ruthhu_Q201103_human_heavyDP_1A1.wiff|RH_20110319_human_heavyDP_1A1_m1|0"

            if file_path.endswith('.raw'):
                # thermo raw file
                raw_file_name = file_path.split('\\')[-1]
                mzml_file_name = f'{raw_file_name[:-4]}.mzML'
            elif file_path.index('.wiff') >= 0:
                # wiff file
                raw_file_name = file_path.split('|')[0].split('\\')[-1]
                mzml_file_name = f'{raw_file_name[:-5]}-{sample_name}.mzML'
            else:
                file_ext = file_path.split('.')[-1]
                raise ValueError(f'.{file_ext} to .mzml mapping is not defined yet')

            rep_infos.append([replicate, sample_file_id, sample_name, file_path, mzml_file_name])

    cols = [
        'replicate_name', 'sample_id', 'sample_name', 'raw_file', 'mzml_file'
    ]
    return pd.DataFrame(rep_infos, columns=cols)


def extract_peak_groups(skyline_tree):
    root = skyline_tree.getroot()

    protein_list = root.findall('peptide_list')
    peak_infos = []
    trans_infos = []

    def extract_transtions(transition_list):
        for transition in transition_list:
            pro_ion = transition.get('fragment_type')+transition.get('fragment_ordinal')
            neutral_loss = float(transition.get('loss_neutral_mass'))
            if neutral_loss > 0:
                pro_ion += f'-{int(neutral_loss)}'
            prod_charge = int(transition.get('product_charge'))
            pre_mz = float(transition.find('precursor_mz').text)
            pro_mz = float(transition.find('product_mz').text)
            
            yield [
                protein_name, seq, proteoform, is_heavy, 
                precursor_charge, prod_charge, pre_mz, pro_mz, pro_ion]

    def extract_peak(precursor_list):
        for peak in precursor_list:
            replicate = peak.get('replicate')
            sample_id = peak.get('file') if 'file' in peak.keys() else f'{replicate}_f0'
            
            rt = np.float32(peak.get('retention_time'))
            st = np.float32(peak.get('start_time'))
            ed = np.float32(peak.get('end_time'))
            dotp = np.float32(peak.get('library_dotp'))

            # some of runs included in the skyline file has not annotated.
            if rt and st and ed:
                yield [
                    sample_id, protein_name, seq, 
                    proteoform, is_heavy, rt, st, ed, dotp]

    for protein in protein_list:
        protein_name = protein.get('label_name')
        peptides = protein.findall('peptide')
        for peptide in peptides:
            seq = peptide.get('sequence')
            proteoform = peptide.get('modified_sequence')
            
            for precursor in peptide.findall('precursor'):
                precursor_charge = int(precursor.get('charge'))
                is_heavy = precursor.get('isotope_label') == 'heavy'
                precursor_list = precursor.find('precursor_results').findall('precursor_peak')
                transition_list = precursor.findall('transition')
                trans_infos.extend(extract_transtions(transition_list))
                peak_infos.extend(extract_peak(precursor_list))

    cols = ['sample_id', 'protein_name', 'sequence', 'modified_sequence',
            'is_heavy', 'RT', 'start_time', 'end_time', 'dotp']
    peak_df = pd.DataFrame(peak_infos, columns=cols)
    
    cols = [
        'protein_name', 'sequence', 'modified_sequence', 
        'is_heavy', 'precursor_charge', 'product_charge', 
        'precursor_mz', 'product_mz', 'product_ion',
    ]

    trans_df = pd.DataFrame(trans_infos, columns=cols)                    

    return peak_df, trans_df

