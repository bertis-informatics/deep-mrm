from pathlib import Path
import subprocess


def convert_to_mzML(ms_file_path):
    # if not msconvert_bin.exists():
    #     raise ValueError(f'Cannot find {msconvert_bin}')
    msconvert_bin = 'msconvert.exe'

    mzml_dir = ms_file_path.parent / f'{ms_file_path.stem}_mzML'
    if not mzml_dir.exists():
        mzml_dir.mkdir()

    # subprocess.run([str(msconvert_bin), mass_spec_path, '--mzML'])
    completed_proc = subprocess.run(
        [str(msconvert_bin), ms_file_path, '--mzML', '-o', f'{mzml_dir}'],
        capture_output=True, 
        #stdout=PIPE, stderr=STDOUT
        )
    
    if completed_proc.returncode != 0:
        raise ValueError(f'{completed_proc.stderr.decode()}')

    s = completed_proc.stdout.decode()
    mzml_files = [Path(x[len('writing output file: '):]) for x in s.split('\n') if x.startswith('writing')]

    return mzml_files
