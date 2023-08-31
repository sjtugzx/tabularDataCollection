# An Automated Tabular Data Collection Method

Code for the Scientific Data Paper "A new method to obtain the Sm-Nd isotope data through automated tabular data collection from geological articles".

**Extracted Data Availability**: [Guo, Zhixin (2023). Sm-Nd data collection.xls. figshare. Dataset.](https://doi.org/10.6084/m9.figshare.24054231.v1)

## Environment

- **Operating System**: Linux System
- **Languages & Libraries**:
  - python==3.9
  - pytorch==1.10.1
  - torchvision==0.11.2
  - torchaudio==0.10.1
- **Tools & Utilities**:
  - cudatoolkit=11.3
  - Cuda 10.1

## Installation

```bash
# Install Conda environment
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create and activate Python environment
conda create -n dde-table python=3.9
conda init bash
conda activate dde-table

# Install dependencies
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 cudnn -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install pdfminer.six xlwt xlrd fitz pdf2image tqdm PyMuPDF opencv-python PyPDF2 pdfplumber
conda install -y -c conda-forge poppler

# Clone Detectron2 source code
git clone https://github.com/facebookresearch/detectron2.git

# Install Detectron2 model
cd detectron2/demo
mkdir models
cd models
wget https://layoutlm.blob.core.windows.net/tablebank/model_zoo/detection/All_X152/model_final.pth
wget https://layoutlm.blob.core.windows.net/tablebank/model_zoo/detection/All_X152/All_X152.yaml

# Copy the extractor source code
cp -r /path_to_ourcode_code/code/extractor .

# Activate the environment and set the path
conda activate dde-table
cd path/to/detectron2/demo

# Copy the filtered literature's PDF files to this directory
# Assuming all files have been copied to ./pdf

# Set the directory for table target recognition, output directory, and GPU card number
# Edit predictor.py to set the CUDA_DEVICE_ID variable to select the GPU card
# Edit bash_table.py to set the PDF_DIR (e.g., ./pdf) and OUTPUT_DIR (e.g., ./detect_output)

# Perform table target recognition; modify CUDA_DEVICE_ID and run repeatedly for multi-card operation
python bash_table.py

# Set the directory for table content parsing
# Edit extract_src/main.py to set PDF_DIR (e.g., ../pdf), DETECT_DIR (e.g., ../detect_output), and MARKS_DIR (e.g., ../marks)

# Perform table content parsing
cd extract_src
python main.py
```

## Appendix

Files in the **Extractor** directory contain the code for tabular data collection. We do not provide the original files for the experiment. Users can follow the instructions to set the File Path for table detection and collection. In this work, we focus on collecting data about the Sm-Nd isotope, following these rules:

For each literature's title and abstract fields, we filter and retain the literature that meets the requirements and its metadata. Finally, we output the metadata of all the literature that meets the requirements to a single JSON file. Regular expressions are mainly used for string matching. For example, for the filtering rule "the title or abstract contains εNd or Sr-Nd or Sr-Nd-Hf or 143Nd/144Nd or Nd isotope or isotopic or TDM", the following regular expression can be used:

```
εNd|Sr-Nd|Sr-Nd-Hf|143Nd/144Nd|Nd isotope|isotopic|TDM
```
