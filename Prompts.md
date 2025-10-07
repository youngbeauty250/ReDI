# Prompt Collection

This document contains the prompt templates used in our ReDI framework for **query decomposition** and **interpretation**.  
They are grouped into three categories: **Decomposition**, **Sparse Interpretation**, and **Dense Interpretation**.

---

## 1. Decomposition Prompt

Used to **decompose a complex query** into multiple focused sub-queries.  

DECOMPOSE_PROMPT = """
You are an expert in search intent analysis.

Please complete the following tasks:

- **Analyze Core Needs**: Extract the primary objective and any implicit requirements of the original query.
- **Design Search Sub-Queries**: Decompose the original query into between three and ten independent sub-queries, each targeting a single information need.

Perform the decomposition according to the following requirements:

#### Decomposition Requirements

1. **Single Focus**: Each sub-query must address one clear information point; avoid combining multiple aspects or topics.
2. **Concise Clarity**: Use direct, declarative sentences; avoid complex or compound sentences.
3. **Independent Retrievability**: Sub-queries should be mutually independent with minimal overlap; if ambiguity exists, split into separate sub-queries.
4. **Quantity Constraint**: Produce at least three and not more than ten sub-queries.

#### Output Format (only output the decomposition results, without any additional analysis or explanations):

Sub_Query_1: "<begin_of_query>...<end_of_query>"
Sub_Query_2: "<begin_of_query>...<end_of_query>"
...
Sub_Query_n: "<begin_of_query>...<end_of_query>"

Query: {query}
"""

## 2. Sparse Interpretation Prompt

Used to generate **interpretations** for sub-queries in **sparse retrieval (BM25)**.

INTERPRETATION_PROMPT_sparse = """
For each Sub-Query, identify its essential information need and generate a concise yet enriched interpretation to better support retrieval. Your interpretation should be guided by the following steps:
1. **Clarify the core intent**: Identify the key concept or problem the sub-query addresses.
2. **Expand semantically and lexically**: Include the singular and plural forms of core terms. Add common derivations (e.g., "-ing", "-ion", "-ed", etc.). Incorporate 3–5 close synonyms or near-synonyms to improve coverage. Consider relevant domain-specific jargon or frequent phrase variants that are likely to appear in relevant documents.
3. **Think step by step**: Reason thoroughly about what supporting information, alternative expressions, or formulations would help capture relevant content from the corpus.

### Input Format:
Sub-queries:  
{intent}

### Output Format:

Interp1: "<|begin_of_interp|>Concise interpretation of Sub_Query_1.<|end_of_interp|>"
Interp2: "<|begin_of_interp|>Concise interpretation of Sub_Query_2.<|end_of_interp|>"
...
Interpn: "<|begin_of_interp|>Concise interpretation of Sub_Query_n.<|end_of_interp|>"

Now, generate interpretations for each provided sub-query accordingly.
"""

## 3. Dense Interpretation Prompt

Used to generate **interpretations** for sub-queries in **dense retrieval (SBERT)**.

INTERPRETATION_PROMPT = """
Instructions:
For each Sub-Query, write a context-rich paraphrase of ~20–30 words that:
1. Restates the query in natural language,
2. Includes related concepts or use cases,
3. Highlights implicit relationships or typical scenarios.

### Input Format:
Sub-queries:  
{intent}

### Output Format:

Interp1: "<begin_of_interp>Concise interpretation of Sub_Query_1.<end_of_interp>"
Interp2: "<begin_of_interp>Concise interpretation of Sub_Query_2.<end_of_interp>"
...
Interpn: "<begin_of_interp>Concise interpretation of Sub_Query_n.<end_of_interp>"

Now, generate interpretations for each provided sub-query accordingly.
"""

