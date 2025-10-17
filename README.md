# Green Skill Forecasting
Research project that implements the entire **pipeline** for cleaning, classifying, and predicting green skills in the Mexican automotive industry.

## How to run
1. Clone the repository. 
```bash
git clone https://github.com/JNArrazola/green_skill_RAG_system
```
2. Create a virtual environment and install the required packages using `pip install -r requirements.txt`.
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```
3. Obtain an OpenAI API key and set it as an environment variable named `OPENAI_KEY`.
```bash
# .env file
OPENAI_KEY="your_openai_api_key"
```
4. Download the datasets folder and paste it in the root directory (not publicly available for the moment).
```txt
green_skill_RAG_system
├── data <-- Download this folder and paste it here
│   ├── embeddings
│   ├── green_jobs_normalized.csv
│   ├── green_jobs_with_titles.csv
│   ├── jan_to_apr_2025_with_languages_cleaned.csv
│   ├── jan_to_apr_2025_with_languages.csv
│   ├── mapping
│   ├── skill_frequencies.csv
│   └── taxonomies
├── env
│   ├── bin
│   ├── include
│   ├── lib
│   ├── lib64 -> lib
│   ├── pyvenv.cfg
│   └── share
├── README.md
├── requirements.txt
├── src
│   ├── 1_preparing_dataset.ipynb
│   ├── 2_generate_embeddings.ipynb
│   ├── 3_green_job_detection_and_normalization.ipynb
│   ├── 4_descriptive_analysis.ipynb
│   └── random.txt
└── test
    ├── test_openai.py
    ├── test.py
    └── text_input_openai.py
```
5. Run the notebooks in the following order:
    1. `1_preparing_dataset.ipynb`
    2. `2_generate_embeddings.ipynb`
    3. `3_green_job_detection_and_normalization.ipynb`
    4. `4_descriptive_analysis.ipynb`

## Notebooks
1. **1_preparing_dataset.ipynb:** Notebook for cleaning and preparing the dataset.
2. **2_generate_embeddings.ipynb:** Notebook for generating embeddings of skills and jobs using OpenAI's API.
3. **3_green_job_detection_and_normalization.ipynb:** Notebook for detecting green jobs and normalizing them to ESCO taxonomy.
4. **4_descriptive_analysis.ipynb:** Notebook for descriptive analysis of the green skills and jobs.

## Definitions
* **Knowledge:** Knowledge means the outcome of the assimilation of information through learning. Knowledge is the body of facts, principles, theories and practices that is related to a field of work or study.
* **Skill/Competence:** Skill means the ability to apply knowledge and use know-how to complete tasks and solve problems

> Source: [European Skills, Competences, Qualifications and Occupations (ESCO)](https://esco.ec.europa.eu/en/about-esco/escopedia/escopedia/knowledge)

## Esco taxonomy
The ESCO taxonomy can be downloaded from [here](https://ec.europa.eu/newsroom/empl/items/741088/en). The data used in this project is from version `1.2.0`.

# Authors
* PhD. Sabur Butt
* Doctor Hector Ceballos
* Joshua Arrazola