# AI-Rational-Medication-System

The official GitHub repository of the paper "Promoting Appropriate Medication UseLeveraging Medical Big Data".
## Abstract
According to the statistics of the World Health Organization (WHO), inappropriate medication has become an important factor affecting the safety of rational medication. In the gray area of medical insurance supervision, such as designated drugstores and designated medical institutions, there are lots of inappropriate medication phenomena about "big prescription for minor ailments". While the traditional Clinical Decision Support System (CDSS) are mostly based on established rules to regulate inappropriate prescriptions, which are not suitable for clinical environments that require intelligent review. In this paper, we model the complex relation among patient, disease and drug based on medical big data to promote the appropriate medication use. More specifically, we first construct the medication knowledge graph based on the historical prescription big data of tertiary hospitals and medical text data. Second, based on the medication knowledge graph, we employ Gaussian Mixture Model (GMM) to group patient population representation as physiological features. For diagnostic features, we employ pre-training word vector BERT to enhance the semantic representation between diagnoses. And in order to reduce adverse drug interactions caused by drug combination, we employ graph convolution network to transform drug interaction information into drug interaction features. Finally, we employ the sequence generation model to learn the complex relationship among patient, disease and drug, and provide an appropriate medication evaluation for prescribing by doctors in small hospitals from two aspects of drug list and medication course of treatment. In this paper, we leverage the MIMIC_III dataset and the dataset of a tertiary hospital in Fujian Province to verify the validity of the model. The results show that our method is more effective than other baseline methods in the accuracy of medication regimen prediction of rational medication. In addition, it has achieved high accuracy in the appropriate medication detection of prescription in small hospitals.

## Getting Started
Before you run the code, you need to create a virtual environment and activate it via the following command:
```
conda env create -f environment.yaml
conda activate venv
```
## Deployment
### Pre-Step: Download Model File
Before setting up the Neo4j database, you need to download the model file from Google Drive. Follow these steps:

1. Visit the following URL: [final_model_DDI_True.model](https://drive.google.com/file/d/1Ftnst83JILo4cngF5rBlkD736hxt2tVA/view?usp=drive_link)
2. Download the file named `final_model_DDI_True.model`.
3. Place the downloaded file in the directory `/data/drugEco/transformer_Top3_treatment/` within your project folder.

### Step 1: Neo4j Database Setup

1. **Install Neo4j Community Edition:**
    - Download Neo4j Community Edition version 3.5.5 from the Neo4j Download Page.
    - Follow the installation instructions provided on the website.
2. **Start the Neo4j Service:**
    - Navigate to the Neo4j installation directory.
    - Start the Neo4j server using the following command:
        
        ```bash
        ./bin/neo4j start
        ```
        
3. **Import Knowledge Graph Data:**
    - Ensure that your CSV files are properly formatted and placed in the **`data`** directory of your Neo4j installation.
    - As an example, consider the "Prescription" folder, which contains multiple CSV files such as **`patient.csv`** and **`patient2diag.csv`**.
    - Use the following command to import entities and their attributes from **`patient.csv`**:
        
        ```bash
        neo4j-admin import --mode=csv --database=graph.db --nodes="data/prescription/patient.csv"
        ```
        
    - Use the following command to import relationships from **`patient2diag.csv`**:
        
        ```bash
        neo4j-admin import --mode=csv --database=graph.db --relationships="data/prescription/patient2diag.csv"
        ```
        
    - Note: Ensure that the Neo4j database is not running when you execute these import commands, as the **`neo4j-admin import`** tool requires the database to be offline during the operation.

### Step 2: Python Environment Setup

1. **Install Python Packages:**
    - Ensure Python 3.7 is installed on your system. You can download it from [Python's official site](https://www.python.org/downloads/release/python-370/).
    - Install required Python packages by running:
        
        ```bash
        pip install -r requirements.txt
        
        ```
        

### Step 3: Start the Development Server

1. **Run the Development Server:**
    - Navigate to the project's root directory.
    - Start the server by executing:
        
        ```bash
        python manage.py runserver 0.0.0.0:3502
        
        ```
        
    - This will launch a development server accessible at http://0.0.0.0:3502/ or http://localhost:3502/.

### Step 4: Access the Application

1. **Use the Application:**
    - Open a web browser.
    - Visit http://localhost:3502/ to interact with the application.

## Reference
If our work is helpful to your research, please cite our paper:
``` latex
@article{Hong2024prompting,
  author       = {Linghong Hong, Shiwang Huang, Guofeng Luo, Xiaohai Cai, Jiaru Wang and Chenhui Yang},
  title        = {Promoting Appropriate Medication UseLeveraging Medical Big Data},
  publisher    = {GitHub},
  year         = {2024},
}
```
