# DeepMRM

DeepMRM is a targeted proteomics data interpretation package using deep-learning. It takes MRM, PRM, or Data-Independent Acquisition (DIA) data and target list as input and outputs peaks of targeted peptides, along with their abundances and quality scores. The targeted endogenous (light) peptides are detected and quantified using their stable isotope labeled (heavy) peptides.




## Installation (without GPUs)
You can use DeepMRM with conda environment or docker. 

### Conda environment
We recommend to run DeepMRM in a conda environment such that your environment for DeepMRM and dependencies are separate from other Python environments.

1. Download and install Anaconda  
   You can download Anaconda from https://www.anaconda.com/download

2. Create a conda environment  
   Open the terminal (Linux or MacOS) or the Anaconda Prompt (Windows).   
   Create a new conda environment, named `deepmrm`, and activate it by running following commands:

   ```sh
   conda create -n deepmrm python=3.9
   conda activate deepmrm
   ```

3. Clone the repository and move to the deep-mrm directory in the terminal.
   ```sh
   git clone https://github.com/bertis-informatics/deep-mrm.git
   ```

4. Install required packages
   ```sh
   pip install -r requirments.txt
   ```

5. Install DeepMRM package 
   ```sh
   conda develop .
   ```
   This will output the deep-mrm directory path.

### Docker
1. Download and install docker desktop (https://www.docker.com/get-started/)
2. Pull DeepMRM docker image  
   ```
   docker pull jungkap/deepmrm:v1.6
   ```


## How to run

### Conda environment
1. Open the terminal (Linux or MacOS) or the Anaconda Prompt (Windows), and activate your deepmrm conda environment created during the installation

2. Run DeepMRM running script with a mass-spec file (mzML) and a target list (csv).   
You can find sample input files in [sample_data](https://github.com/bertis-informatics/deep-mrm/tree/main/sample_data) directory.  
   For example, if you are in deep-mrm directory and want to run DeepMRM for sample data, then run the following command:

    ```sh
    python deepmrm/predict/make_prediction.py \
               -target ./sample_data/sample_target_list.csv \
               -input ./sample_data/sample_mrm_data.mzML
    ```

    Syntax
    
    * -input: string path for input mzml file
    * -target: string path for target csv file. The csv file should contain following columns:
        * `peptide_id`: unique ID for each targeted peptide (heavy and light transitions are grouped by peptide_id)
        * `precursor_mz`: m/z values for targeted precursor ions
        * `product_mz`: m/z values for targeted fragment ions
        * `is_heavy`: flag indicating light or heavy peptides (TRUE or FALSE)

            ---------------------------------------------
            Example of target list

            |peptide_id|precursor_mz|product_mz|is_heavy|
            |----------|------------|----------|--------|
            |PDAC0008|798.36005|966.4759|FALSE|
            |PDAC0008|798.36005|1210.596|FALSE|
            |PDAC0008|798.36005|1472.6759|FALSE|
            |PDAC0008|800.36676|972.496|TRUE|
            |PDAC0008|800.36676|1216.596|TRUE|
            |PDAC0008|800.36676|1478.696|TRUE|       
    * -tolerance: int, default=10. Ion match tolerance (in PPM). This is ignored in the case of MRM data.

### Docker
You can run DeepMRM through docker with following command. You should specify  {local_dir_for_input_data}, {target_csv_file_name}, and {mzml_file_name} before running.  
```
docker run -v {local_dir_for_input_data}:/input_data jungkap/deepmrm \
            -target /input_data/{target_csv_file_name} \
            -input /input_data/{mzml_file_name}
```
For example, you can run DeepMRM for sample data in the deepmrm directory as follows.   
In Windows, you need to replace forward slash with backslash in the local directory path.
```
docker run -v ./sample_data:/input_data jungkap/deepmrm:v1.6 -target /input_data/sample_target_list.csv -input /input_data/sample_mrm_data.mzML
```
   

## License
Copyright©2022 [Bertis Inc.](http://bertis.com/) All rights reserved.

DeepMRM package is freely available for academic research, non-profit or educational purposes only under [DeepMRM-LICENSE](https://github.com/bertis-informatics/deep-mrm/blob/main/DeepMRM-LICENSE.txt). For commercial use of DeepMRM, please contact deepmrm@bertis.com.
