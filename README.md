# Innomatics Research Labs RAG Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) conversational assistant designed specifically for **Innomatics Research Labs**[cite: 1, 2]. This intelligent chatbot queries a dedicated knowledge base derived from official institute policies, course syllabi, campus locations, fee structures, and career support services.

---

## 🚀 Key Features

* **Strict Policy Adherence:** Programmed to answer questions exclusively from official knowledge base records without inventing legal interpretations or unauthorized policy exceptions[cite: 1].
* **Comprehensive Domain Coverage:** Access to detailed institutional details including Data Science, Full Stack, Generative AI, and Agentic AI curricula[cite: 3, 8, 10, 11, 12].
* **Multi-Campus Directory:** Quick retrieval of official branch addresses, contact numbers, and support emails across Hyderabad, Pune, and Bengaluru.
* **Dynamic PDF Knowledge Base:** Powered by a structured suite of official reference documents covering placements, certifications, fee structures, and admission workflows[cite: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].

---

## 📂 Repository Structure

```text
Innomatics-Research-Labs-RAG-Assistant/
│
├── data/                       # Official Knowledge Base PDF Documents
│   ├── 01_institute_information.pdf
│   ├── 02_admission_process.pdf
│   ├── 03_fee_structure.pdf
│   ├── 04_placement_information.pdf
│   ├── 05_certification_information.pdf
│   ├── 06_projects_and_internship.pdf
│   ├── 07_frequently_asked_questions.pdf
│   ├── 08_contact_information.pdf
│   └── 04_Advanced_Generative_AI_Syllabus.pdf
│
├── app.py                      # Main Application / RAG Streamlit Interface
├── requirements.txt            # Python Dependencies
├── .gitignore                  # Git Ignore File (excludes virtual envs & local vector stores)
└── README.md                   # Project Documentation
