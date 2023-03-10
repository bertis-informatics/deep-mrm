# DeepMRM

DeepMRM is a targeted proteomics data interpretation package using deep-learning. It takes MRM, PRM, or Data-Independent Acquisition (DIA) data and target list as input and outputs peaks of targeted peptides, along with their abundances and quality scores. The targeted endogenous (light) peptides are detected and quantified using their stable isotope labeled (heavy) peptides.

## Installation (without GPUs)

DeepMRM was tested on Ubuntu 20.04 and Windows 10. 
1. Install python v3.9+ from https://www.python.org/downloads/
2. Clone the repository, and move to the deep-mrm directory in a terminal
   ```sh
   git clone https://github.com/bertis-informatics/deep-mrm.git
   ```
3. Create a virtual environment, and activate it
    ```sh
   python -m venv env
   .\env\Scripts\activate
   ```
4. Install required packages
   ```sh
   pip install -r requirments.txt
   ```
5. Install DeepMRM package
    ```sh
   python setup.py install
   ```


## How to run
1. Open a terminal and activate the virtual environment created during installation

2. Run DeepMRM running script with a mass-spec file (mzML) and a target list (csv). You can find sample input files in [sample_data](https://github.com/bertis-informatics/deep-mrm/tree/main/sample_data) directory.

    ```sh
    python deepmrm/predict/make_prediction.py -model ./models/DeepMRM_Model.pth -data_type MRM -target C:\sample_data\sample_target_list.csv -input c:\sample_data\sample_mrm_data.mzML -tolerance 100
    ```

    Syntax
    * -mode: model file path
    * -data_type: data type, MRM/PRM/DIA
    * -tolerance: ion match tolerance (in PPM)
    * -target: comma-seprated text file for target list, which should contain following columns:
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

## License
Copyright??2022 [Bertis Inc.](http://bertis.com/) All rights reserved.

DeepMRM package is freely available for academic research, non-profit or educational purposes only under [DeepMRM-LICENSE](https://github.com/bertis-informatics/deep-mrm/blob/main/DeepMRM-LICENSE.txt). For commercial use of DeepMRM, please contact deepmrm@bertis.com.
