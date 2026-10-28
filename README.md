# 🏡 Old Age Home Recommendation System

### AI-Powered Elderly Care Discovery and Recommendation Platform

An intelligent, content-based recommendation system that helps users discover suitable old-age homes across India based on location, care requirements, facility type, gender eligibility, management type, and personal preferences.

The project combines **TF-IDF vectorization, cosine similarity, rule-based filtering, and an interactive Streamlit frontend** to simplify the process of finding and comparing old-age homes.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Dataset Description](#-dataset-description)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Recommendation Methodology](#-recommendation-methodology)
- [Match Score Calculation](#-match-score-calculation)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [Rating and Feedback](#-rating-and-feedback)
- [Model Evaluation](#-model-evaluation)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [References](#-references)

---

## 🌟 Overview

Finding an appropriate old-age home can be challenging because information about residential care facilities is distributed across directories, government records, and individual websites.

This project develops a recommendation platform that processes an old-age-home directory and ranks listings according to user-selected requirements and descriptive preferences.

Users can interact with the application to:

- Select a preferred state and district.
- Filter homes by facility category, care type, gender eligibility, and management type.
- Enter natural-language preferences.
- View ranked recommendations and available facility details.
- Inspect match scores and contact information.
- Rate recommendations and provide optional feedback.
- Download recommendation results as a CSV file.

The system is designed as a decision-support tool for discovering potential care facilities. It does not independently verify service quality, medical suitability, fees, or current availability.

## ❗ Problem Statement

Families searching for old-age homes may face several challenges:

- Information is spread across multiple sources.
- Comparing facilities manually is time-consuming.
- Important attributes such as capacity and contact information may be incomplete.
- Generic directories do not necessarily support personalized filtering and ranking.
- Users may struggle to identify which facilities best match their requirements.

**Proposed solution:** Develop a content-based recommendation system that filters facility listings using explicit requirements and ranks the remaining candidates using textual similarity.

## 🎯 Objectives

1. Process and clean old-age-home listing data.
2. Explore geographic coverage and data completeness.
3. Develop a content-based recommendation model.
4. Apply hard eligibility filters before ranking candidates.
5. Provide an interactive interface using Streamlit.
6. Collect user ratings for preliminary recommendation evaluation.
7. Highlight missing or unverified information instead of assuming it is available.

## ✨ Key Features

### 1. Personalized Search

Users can specify their preferred location, facility category, care type, gender eligibility, and management type.

### 2. Content-Based Recommendation

The model compares user-entered preferences with descriptive information about each home using TF-IDF and cosine similarity.

### 3. Hard Constraint Filtering

Listings that do not satisfy selected filters are excluded from the recommendation results.

### 4. Match Score

Each recommendation receives a numerical ranking score based on text similarity and contact-information completeness.

### 5. Facility Information

The interface displays available details such as:

- Home name
- State and district
- Address
- Facility category
- Care type
- Gender eligibility
- Management type
- Reported bed capacity
- Phone number and email
- Source information, when available

### 6. User Rating System

Users can rate recommendations on a scale of 1–5 and optionally submit feedback. Ratings are saved locally for subsequent analysis.

### 7. CSV Export

Users can download the current recommendation results for comparison or further analysis.

### 8. Evaluation Pipeline

The evaluation script calculates preliminary ranking and feedback metrics using collected ratings.

---

## 📊 Dataset Description

### Primary Dataset

**File:** `All_India_Old_Age_Homes.xlsx`

**Worksheet:** `Master_Data`

The project uses this workbook as its primary facility directory because it provides listing-level attributes that support location filtering, facility matching, and content-based recommendation.

The inspected version contains:

- **518 records**
- **20 source columns**

Available fields include home name, state, district, locality, address, PIN code, reported bed capacity, facility type, management type, gender eligibility, care type, contact information, and source-related fields.

### Dataset Preprocessing

The preprocessing pipeline includes:

1. Text normalization and whitespace cleaning.
2. Missing-value identification.
3. Numeric conversion and capacity validation.
4. Duplicate detection and removal of exact duplicate rows.
5. Standardization of categorical attributes.
6. Creation of missing-value and contact-completeness flags.
7. Construction of descriptive text for recommendation.
8. Generation of a cleaned dataset for model training.

Missing values are not automatically replaced with fabricated information. Zero or missing capacity is flagged for verification.

### Additional Datasets Reviewed

Other downloaded sources included:

- Integrated Programme for Senior Citizens (IPSrC) government-program data.
- A smaller CSV extract of Integrated Programme for Older Persons records.

These datasets can support future enrichment and cross-validation. The current recommendation model is based on `All_India_Old_Age_Homes.xlsx`.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading, preprocessing, and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| Joblib | Saving and loading the fitted model bundle |
| Streamlit | Interactive web application |
| Matplotlib | Supporting data visualization |
| OpenPyXL | Reading Excel datasets |

---

## 🏗️ System Architecture

The system follows a simple, modular recommendation pipeline:

```text
All_India_Old_Age_Homes.xlsx
              |
              v
     Data Preprocessing
              |
              v
    Feature Engineering
              |
              v
      TF-IDF Vectorizer
              |
              v
      Fitted Model Bundle
              |
              v
       User Preferences
              |
              v
      Hard Constraint Filters
              |
              v
       Cosine Similarity
              |
              v
    Ranked Recommendations
              |
              v
       Streamlit Frontend
              |
       +------+------+
       |             |
       v             v
  User Ratings    CSV Export
       |
       v
 Evaluation Report
```

## 🧠 Recommendation Methodology

### 1. Content-Based Filtering

The model uses descriptive information associated with each home, including its name, location, facility type, care type, management type, gender eligibility, address, and remarks when available.

### 2. TF-IDF Vectorization

Term Frequency–Inverse Document Frequency (TF-IDF) transforms text into numerical feature vectors. The fitted vectorizer uses unigrams and bigrams, with a maximum vocabulary size of 8,000 features.

### 3. Cosine Similarity

Cosine similarity measures the similarity between the user's text query and the text representation of each eligible home.

\[
S=\frac{A\cdot B}{\|A\|\|B\|}
\]

Here, \(A\) represents the query vector and \(B\) represents a home's feature vector.

### 4. Hard Constraint Filtering

Before ranking, the system filters candidates according to the selected state, district, facility type, care type, gender eligibility, and management type.

### 5. Ranking

The eligible homes are ranked using text similarity, with an optional contact-completeness contribution.

Collaborative filtering is not implemented because the original dataset does not provide a reliable user–home interaction history.

## 📐 Match Score Calculation

The current implementation calculates:

\[
\text{Match Score}=0.85S+0.15C
\]

Where:

- \(S\) = TF-IDF cosine similarity.
- \(C\) = contact-completeness score between 0 and 1.

The contact-completeness score is calculated as:

\[
C=0.5P+0.2E+0.3A
\]

Where:

- \(P=1\) if a phone number is available; otherwise 0.
- \(E=1\) if an email address is available; otherwise 0.
- \(A=1\) if an address is available; otherwise 0.

**Important:** The match score is a ranking signal, not a probability, a facility-quality rating, or a guarantee of suitability. Contact completeness indicates how much contact information is present, not the quality of care.

---

## 📁 Project Structure

```text
old_age_home_recommender/
│
├── app.py
│
├── train_model.py
│
├── evaluate.py
│
├── requirements.txt
│
├── README.md
│
├── data/
│   ├── All_India_Old_Age_Homes.xlsx
│   └── user_ratings.csv
│
├── artifacts/
│   ├── content_model.joblib
│   ├── homes_cleaned.csv
│   └── evaluation_report.csv
│
└── src/
    └── recommender.py
```

Generated artifacts appear after the corresponding scripts run. The evaluation report is generated after valid ratings have been collected.

## ⚙️ Installation and Setup

### Prerequisites

Install:

- Python 3.10 or a compatible Python version.
- pip.
- A terminal or PowerShell.
- A browser for the Streamlit application.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd old_age_home_recommender
```

Replace the placeholder with your actual repository URL.

### Step 2: Create a Virtual Environment

On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Add the Dataset

Place `All_India_Old_Age_Homes.xlsx` inside the `data/` directory.

### Step 5: Train the Model

```bash
python train_model.py
```

To specify a different dataset location:

```bash
python train_model.py --input "/path/to/All_India_Old_Age_Homes.xlsx"
```

Training fits the TF-IDF vectorizer, constructs the text feature matrix, and saves the cleaned dataset and model bundle.

## 🚀 Running the Application

Launch the Streamlit frontend:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

### Typical User Workflow

1. Select the required location and facility attributes.
2. Enter optional descriptive preferences.
3. Choose the number of recommendations.
4. Click **Find suitable homes**.
5. Review the results, scores, and missing-information warnings.
6. Rate relevant recommendations if desired.
7. Download the results as a CSV file.

## ⭐ Rating and Feedback

When a user submits a rating, the application appends it to:

`data/user_ratings.csv`

The rating log stores fields such as:

- Home identifier and name
- Rating from 1 to 5
- Preference query
- Optional feedback notes
- Recommendation rank
- Recommendation score
- UTC timestamp

The ratings are stored locally on the machine running the application. They are not automatically uploaded to a database or cloud service.

## 📈 Model Evaluation

After collecting ratings, run:

```bash
python evaluate.py --k 5
```

The script generates:

`artifacts/evaluation_report.csv`

The evaluation uses the following preliminary measures:

| Metric | Purpose |
|---|---|
| Mean rating | Average user-assigned rating |
| Relevant share | Proportion of ratings of 4–5 |
| Precision@K | Relevant rated items among the evaluated top-K items |
| Recall@K | Relevant judged items retrieved among the evaluated items |
| NDCG@K | Ranking quality using graded ratings |

Ratings of 4–5 are treated as relevant, ratings of 1–2 as not relevant, and ratings of 3 as neutral.

**Evaluation limitation:** These metrics are calculated from the rated homes available in the log. Unrated homes are unknown, not automatically irrelevant. Reliable claims about recommendation performance require a more complete test set with expert- or user-judged relevance for candidate homes.

## ⚠️ Limitations

- The dataset is a directory, not a live inventory of available beds.
- Fees and current vacancies are not reliably available as verified fields.
- Medical support, accessibility, dementia care, and service quality cannot be assumed from textual similarity.
- Missing phone numbers, email addresses, or capacity information may require direct confirmation.
- Contact completeness may affect ranking but does not establish suitability.
- The system does not currently use collaborative filtering or a trained reinforcement-learning agent.
- The model has not yet been validated through a representative user study.

## 🔮 Future Enhancements

1. **Geographic recommendations:** Use validated addresses and geocoding to rank homes by distance.
2. **Verified affordability matching:** Add confirmed monthly fees and free/subsidized accommodation eligibility.
3. **Service verification:** Collect confirmed medical support, nursing availability, accessibility, and specialized-care attributes.
4. **Improved ranking:** Combine textual similarity with verified feature-level preference scores.
5. **Better evaluation:** Build a labeled relevance dataset and benchmark Precision@K, Recall@K, and NDCG@K.
6. **Database integration:** Store facility records, user preferences, and ratings in PostgreSQL.
7. **Collaborative filtering:** Explore user–item recommendation models after collecting sufficient reliable interaction data.
8. **Deployment:** Host the Streamlit application with suitable persistent storage and privacy controls.

## 📚 References

1. Elderly Care India — List of Old Age Homes in India  
   https://elderlycareindia.org/list-of-old-age-homes-in-india/

2. AIKosh — Integrated Programme for Older Persons, Government of India Grant Data  
   https://aikosh.indiaai.gov.in/web/datasets/details/integrated_programme_for_older_persons_goi_grant_data.html

3. data.gov.in — Integrated Programme for Older Persons  
   https://www.data.gov.in/resource/integrated-programme-older-persons-goi-grant

4. Department of Social Justice and Empowerment — Integrated Programme for Senior Citizens  
   https://www.dosje.gov.in/schemes-and-services/integrated-programme-for-senior-citizens-ipsrc/

5. NEMA Care — Old Age Homes in India: The Complete End-to-End Guide  
   https://www.nemacare.com/post/old-age-homes-in-india-the-complete-end-to-end-guide

6. Elkin — Senior Care Resources  
   https://www.elkin.co.in/

---

## Contributors

This project was developed as a collaborative team effort:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/KanishkaGole">
        <img src="https://github.com/KanishkaGole.png" width="100px;" alt="Kanishka Gole"/>
        <br />
        <sub><b>Kanishka Gole</b></sub>
      </a>
      <br />
      <a href="https://github.com/KanishkaGole/Old-Age-Home-Recommendation-System/commits?author=KanishkaGole" title="Code">💻</a>
    </td>
    
  </tr>
</table>

---

## 📄 License

Choose an appropriate license before publishing the repository. Also verify the redistribution terms of the source dataset and its associated listings before including the workbook in a public repository.
