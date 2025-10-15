# Potential Talents: Retrieval for Candidate Job Titles (NLP)

### Industry: 
Human Resources - Talent sourcing

### Context: 
In the talent sourcing industry, connecting the right candidate to the right opportunity is a complex challenge. Recruiters must not only understand a client’s technical requirements but also recognize what makes a candidate truly stand out for a given role. Traditional keyword-based search often fails to capture the nuances of job titles, skills, and role similarities,  especially in the fast-evolving technology sector.

## Objective:
Build a robust, intelligent retrieval system capable of identifying and ranking semantically similar job titles and candidate profiles. 
Exploring different approaches, from most elementary like using TF-IDF to more evolved using SBERT, LLM Fine Tuning, RAG with FAISS indexing and consulting big LLM through API.

## What we include in this project? 
This work involved a lot of different techniques in NLP, from simpler approaches to more evolved. Applying different combination of techniques, metrics, solution design and two final efficient approaches prepared for implementation: 
  - a Fine Tuned LLM based on LLaMA 3.2 3B (covered in the project part 5)
  - a RAG (indexed using FAISS) + a big LLM using API (covered in the project part 6)

---

### Project Highlights  

- **Full retrieval ladder:** `TF-IDF → Word2Vec (SGNS) → SBERT + FAISS → LLM re-ranking`  
- **Custom mini-Word2Vec** embedding trained from scratch on domain text  
- **Local LLM inference** on GPU: <img src="./sup_imgs/phi-3-mini.png" width="150" style="vertical-align: middle;"/> <img src="./sup_imgs/gemma2.webp" width="170" style="vertical-align: middle;"/> <img src="./sup_imgs/Qwen_Logo.svg.png" width="140" style="vertical-align: middle;"/> 
  <img src="./sup_imgs/LLaMA3.png" alt="DeepSeek Logo" width="140" style="vertical-align: middle;"/>
- **External API LLMs:** 
  <img src="./sup_imgs/moonshot_AI.png" width="130" style="vertical-align: middle;"/> Kimi K2, 
  <img src="./sup_imgs/DeepSeek_logo.svg.png" alt="DeepSeek Logo" width="160" style="vertical-align: middle;"/> deepseek-chat,
  <img src="./sup_imgs/Grok-feb-2025-logo.svg.png" width="140" style="vertical-align: middle;"/> Grok 4 fast (xAI),
  <img src="./sup_imgs/ChatGPT-Logo.svg.webp" width="50" style="vertical-align: middle;"/> gpt-4o
- **Instruction Fine-Tuned LLaMA 3.2 3B** for candidate scoring  
- **RAG with FAISS (IVF index)** centroid-based ANN tuned (`nlist`, `nprobe`) for **high recall** and **3–8× speedup** over exact search  
- **Comprehensive evaluation:** Recall@k, nDCG@k, and latency, plus qualitative checks on Data Science, ML Engineering, Backend, and PM queries  


---

## RAG Solution Architecture (Part 6)

```
Query
│
├─► SBERT encoder (all-mpnet-base-v2) → L2-normalized vectors
│
├─► FAISS IVF (centroid-based ANN)
│ • nlist=32, nprobe=16 (balanced recall/speed)
│ • retrieve top-N candidate titles
│
├─► Prompt composer (compact context)
│ • formats top-N with [index], score, title
│
└─► LLM re-ranker (OpenAI, API accessed)
```

**Performance snapshot**

- **Retrieval latency:** ~**0.2–0.3 ms/query** (IVF, cosine via inner product on normalized vectors)  
- **Bottleneck:** LLM inference time (seconds), not vector search

---

## Repository Structure

### Notebooks

- **[Part 1 - titleCapstone](.Potential_Talents_part1.ipynb)**  
  EDA + TF-IDF + Cosine Search. Explores and cleans the dataset, builds a manual and scikit-learn TF-IDF representation, and validates cosine similarity for job-title retrieval. 
- **[Part 2 - titleCapstone](.Potential_Talents_part2.ipynb)**  
  Word2Vec Embeddings. Trains a custom skip-gram model on job titles, visualizes the learned vector space, and compares semantic proximity between technical roles. 
- **[Part 3 - titleCapstone](.Potential_Talents_part3.ipynb)**  
  Sentence Transformers (SBERT). Encodes job titles using all-mpnet-base-v2 and performs semantic search with cosine similarity, showing major accuracy gains over classical embeddings.
- **[Part 4 - titleCapstone](.Potential_Talents_part4.ipynb)**  
  Large-Scale Evaluation and Ranking. Benchmarks SBERT retrieval using Recall@k and nDCG@k. Prepares labeled datasets for supervised fine-tuning.
- **[Part 5 - titleCapstone](.Potential_Talents_part5.ipynb)**    
  Instruction Fine-Tuning Dataset. Generates chat-formatted JSONL (train/val/test) pairs with numeric similarity labels, preparing data for model fine-tuning.
- **[Part 5t - titleCapstone](.Potential_Talents_part5t.ipynb)**    
  LoRA Fine-Tuning (Training). Fine-tunes meta-llama/Llama-3.2-3B-Instruct using LoRA adapters to predict similarity scores (0–100) for query-title pairs on a single GPU.
- **[Part 6 - titleCapstone](.Potential_Talents_part1.ipynb)**    
  RAG + FAISS Retrieval. Implements a centroid-based FAISS index (IVF) for fast ANN search, integrates an API-based LLM for re-ranking, and evaluates recall, nDCG, and latency.


### Folder Structure:

```
Potential_Talents/
│   .env                                 # API urls and keys
│   .gitignore
│   README.md
│   tools_llm_ift.py                     # Toolkit for scoring, fine-tuning data prep, and LLM ranking
│   pairwise_llm_dataset_ready.csv       # Final labeled dataset (query-title pairs + scores)
│   pairwise_llm_dataset_ready_bkp.csv   # Backup of final dataset
│   pairwise_llm_dataset_skeleton.csv    # Base skeleton before scoring
│
│   Potential_Talents_part1.ipynb        # EDA, TF-IDF, and cosine search baseline
│   Potential_Talents_part2.ipynb        # Word2Vec embeddings and semantic similarity
│   Potential_Talents_part3.ipynb        # SBERT embeddings and semantic retrieval
│   Potential_Talents_part4.ipynb        # Evaluation and dataset preparation
│   Potential_Talents_part5.ipynb        # Instruction dataset generation (JSONL)
│   Potential_Talents_part5t.ipynb       # LoRA fine-tuning for LLaMA 3.2 3B
│   Potential_Talents_part6.ipynb        # RAG + FAISS + API LLM re-ranking
│
├── data/
│   ├── potential_talents.csv            # Original dataset of job titles
│   ├── text8 / text8.zip                # Corpus used for mini Word2Vec training
│   └── test_Official_word2vec.ipynb     # Validation notebook for Word2Vec embedding
│
├── ft_data/
│   ├── llama_pairwise_train.jsonl       # Fine-tuning training split
│   ├── llama_pairwise_val.jsonl         # Fine-tuning validation split
│   └── llama_pairwise_test.jsonl        # Fine-tuning test split
│
├── outputs/
│   ├── sbert_ranking_output.csv         # SBERT semantic ranking results
│   │
│   ├── llm/                             # LLM re-ranking results from different APIs
│   │   ├── llm_top10__llama-3.2-3b-instruct-pairwise__all_queries.csv
│   │   ├── llm_top10__gpt-4o__pairwise__all_queries.csv
│   │   ├── llm_top10__gemma-2-2b-it__listwise__all_queries.csv
│   │   ├── llm_top10__phi3_mini_4k__listwise__all_queries.csv
│   │   ├── llm_top10__qwen2.5-3b-instruct__listwise__all_queries.csv
│   │   ├── llm_top10__deepseek-chat__pairwise__all_queries.csv
│   │   ├── llm_top10__grok-4-fast__pairwise__all_queries.csv
│   │   └── llm_top10__kimi-k2-0905-preview__listwise__all_queries.csv
│   │
│   └── rag_index/                       # RAG FAISS index data
│       ├── embeddings.npy
│       ├── flatip.index                 # Exact search (IndexFlatIP)
│       ├── ivf_nlist32.index            # Centroid-based ANN index (IVF)
│       └── titles.parquet               # Serialized job titles
│
└── sup_imgs/                            # Supplementary images for documentation or reports


```

(this is the content from tree /F that we need to consider and adapt to the previous layout)
```
E:\Devs\pyEnv-1\Apziva\Potential_Talents_-_Yc1Y0PqXJsbGtBoy>tree /F
Folder PATH listing for volume Data
Volume serial number is E29F-7F6B
E:.
│   .env
│   .gitignore
│   pairwise_llm_dataset_ready.csv
│   pairwise_llm_dataset_ready_bkp.csv
│   pairwise_llm_dataset_skeleton.csv
│   Potential_Talents_part1.ipynb
│   Potential_Talents_part2.ipynb
│   Potential_Talents_part3.ipynb
│   Potential_Talents_part4.ipynb
│   Potential_Talents_part5.ipynb
│   Potential_Talents_part5t.ipynb
│   Potential_Talents_part6.ipynb
│   README.md
│   tools_llm_ift.py
│
├───checkpoints
│       sgns_text8.pt
│
├───data
│       potential_talents.csv
│       test_Official_word2vec.ipynb
│       text8
│       text8.zip
│
├───ft_data
│       llama_pairwise_test.jsonl
│       llama_pairwise_train.jsonl
│       llama_pairwise_val.jsonl
│
├───offline_models
│   ├───llama3b_instruct_ft_adapter
│   │       adapter_config.json
│   │       adapter_model.safetensors
│   │       chat_template.jinja
│   │       README.md
│   │       special_tokens_map.json
│   │       tokenizer.json
│   │       tokenizer_config.json
│   │       training_args.bin
│   │
│   ├───llama3b_instruct_ft_base
│   │       chat_template.jinja
│   │       config.json
│   │       generation_config.json
│   │       model-00001-of-00002.safetensors
│   │       model-00002-of-00002.safetensors
│   │       model.safetensors.index.json
│   │       special_tokens_map.json
│   │       tokenizer.json
│   │       tokenizer_config.json
│   │
│   └───llama3b_instruct_ft_merged
├───outputs
│   │   sbert_ranking_output.csv
│   │
│   ├───llm
│   │       llm_top10__deepseek-chat__pairwise__pairwise__all_queries.csv
│   │       llm_top10__gemma-2-2b-it__listwise__all_queries.csv
│   │       llm_top10__gpt-4o__pairwise__pairwise__all_queries.csv
│   │       llm_top10__grok-4-fast__pairwise__all_queries.csv
│   │       llm_top10__kimi-k2-0905-preview__listwise__all_queries.csv
│   │       llm_top10__llama-3.2-3b-instruct-pairwise__all_queries.csv
│   │       llm_top10__phi3_mini_4k__listwise__all_queries.csv
│   │       llm_top10__qwen2.5-3b-instruct__listwise__all_queries.csv
│   │
│   └───rag_index
│           embeddings.npy
│           flatip.index
│           ivf_nlist32.index
│           titles.parquet
│
├───sup_imgs

```
