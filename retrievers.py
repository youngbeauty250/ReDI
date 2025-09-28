import os.path
import re
import time
import torch
import json
import cohere
import numpy as np
import vertexai
import pytrec_eval
import tiktoken
import voyageai
from tqdm import tqdm,trange
import torch.nn.functional as F
from gritlm import GritLM
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from InstructorEmbedding import INSTRUCTOR

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

def cut_text(text,tokenizer,threshold):
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text,tokenizer,threshold=6000):
    token_ids = tokenizer.encode(text)
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text

def get_embedding_google(texts,task,model,dimensionality=768):
    success = False
    while not success:
        try:
            new_texts = []
            for t in texts:
                if t.strip()=='':
                    print('empty content')
                    new_texts.append('empty')
                else:
                    new_texts.append(t)
            texts = new_texts
            inputs = [TextEmbeddingInput(text, task) for text in texts]
            kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
            embeddings = model.get_embeddings(inputs, **kwargs)
            success = True
        except Exception as e:
            print(e)
    return [embedding.values for embedding in embeddings]

def get_embedding_openai(texts, openai_client,tokenizer,model="text-embedding-3-large"):
    texts =[json.dumps(text.replace("\n", " ")) for text in texts]
    success = False
    threshold = 6000
    count = 0
    cur_emb = None
    exec_count = 0
    while not success:
        exec_count += 1
        if exec_count>5:
            print('execute too many times')
            exit(0)
        try:
            emb_obj = openai_client.embeddings.create(input=texts, model=model).data
            cur_emb = [e.embedding for e in emb_obj]
            success = True
        except Exception as e:
            print(e)
            count += 1
            threshold -= 500
            if count>4:
                print('openai cut',count)
                exit(0)
            new_texts = []
            for t in texts:
                new_texts.append(cut_text_openai(text=t, tokenizer=tokenizer,threshold=threshold))
            texts = new_texts
    if cur_emb is None:
        raise ValueError("Fail to embed, openai")
    return cur_emb

TASK_MAP = {
    'biology': 'Biology',
    'earth_science': 'Earth Science',
    'economics': 'Economics',
    'psychology': 'Psychology',
    'robotics': 'Robotics',
    'stackoverflow': 'Stack Overflow',
    'sustainable_living': 'Sustainable Living',
}

def add_instruct_concatenate(texts,task,instruction):
    return [instruction.format(task=task)+t for t in texts]

def add_instruct_list(texts,task,instruction):
    return [[instruction.format(task=task),t] for t in texts]

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_scores(query_ids,doc_ids,scores,excluded_ids):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores


@torch.no_grad()
def retrieval_sf_qwen_e5(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='sf':
        tokenizer = AutoTokenizer.from_pretrained('salesforce/sfr-embedding-mistral')
        model = AutoModel.from_pretrained('salesforce/sfr-embedding-mistral',device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='qwen2':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='e5':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model = model.eval()
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    batch_size = kwargs.get('encode_batch_size',1)

    doc_emb = None
    cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        # already exists so we can just load it
        doc_emb = np.load(cache_path, allow_pickle=True)
    
    for start_idx in trange(0,len(documents),batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue

        batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        # doc_emb[start_idx] = embeddings
        doc_emb = embeddings if doc_emb is None else np.concatenate((doc_emb, np.array(embeddings)), axis=0)

        # save the embeddings every 1000 iters, you can adjust this as needed
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)
        
    np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:",doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
        query_emb += embeddings
    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

@torch.no_grad()
def retrieval_sbert_bge(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id == "bge":
        model_path = "BAAI/bge-large-en-v1.5"
    else:
        model_path = "sentence-transformers/all-mpnet-base-v2"

    model = SentenceTransformer(model_path)
    batch_size = kwargs.get('batch_size',128)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_emb)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

@torch.no_grad()
def retrieval_sbert_bge_fusion_desc(
    queries,
    query_ids,
    documents,
    doc_ids,
    task,
    instructions,
    model_id,
    cache_dir,
    excluded_ids,
    long_context,
    ground_truth=None,
    **kwargs
):

    # ========== Default Setting ==========
    embed_method  = kwargs.get("embed_method", "separate")   # "joint" or "separate"
    desc_weight   = kwargs.get("desc_weight", 0.5)           # 0~1 
    fusion_method = kwargs.get("fusion_method", "sum")       # "sum" or "max"
    # =============================

    # -- 1. Load Embedding Model --
    if model_id == "bge":
        model_path = "BAAI/bge-large-en-v1.5"
    else:
        model_path = "sentence-transformers/all-mpnet-base-v2"

    model = SentenceTransformer(model_path)

    # -- 2. Load Doc Embedding Cache --
    batch_size   = kwargs.get("batch_size", 128)
    doc_emb_dir  = os.path.join(cache_dir, "doc_emb", "sbert", task, f"long_{long_context}_{batch_size}")
    os.makedirs(doc_emb_dir, exist_ok=True)
    doc_emb_file = os.path.join(doc_emb_dir, "docs.npy")
    if os.path.isfile(doc_emb_file):
        doc_emb = np.load(doc_emb_file, allow_pickle=True)
    else:
        doc_emb = model.encode(
            documents,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True
        )
        np.save(doc_emb_file, doc_emb)

    # -- 3. Regex --
    unit_pattern = re.compile(
        r'Sub_Query_\d+:\s*"<begin_of_query>\s*(.*?)\s*<end_of_query>"\s*'
        r'Desc\d+:\s*"<begin_of_desc>\s*(.*?)\s*<end_of_desc>"',
        flags=re.DOTALL
    )

    # -- 4. Unit Embedding Cache Dir --
    if embed_method == "joint":
        joint_emb_dir = os.path.join(cache_dir, "joint_emb", task, f"long_{long_context}_{batch_size}")
        os.makedirs(joint_emb_dir, exist_ok=True)
    else:
        q_emb_dir = os.path.join(cache_dir, "q_emb", task, f"long_{long_context}_{batch_size}")
        d_emb_dir = os.path.join(cache_dir, "d_emb", task, f"long_{long_context}_{batch_size}")
        os.makedirs(q_emb_dir, exist_ok=True)
        os.makedirs(d_emb_dir, exist_ok=True)

    # -- 5. Results --
    fused_scores     = {}
    per_subq_hits    = {}
    per_subq_docs    = {}
    fused_hit_counts = {}

    for qid, big in tqdm(zip(query_ids, queries), total=len(queries), desc="Dense Fusion Desc"):
        q_texts, d_texts = [], []
        for m in unit_pattern.finditer(big):
            q_texts.append(m.group(1).strip())
            d_texts.append(m.group(2).strip())
        if not q_texts:
            q_texts, d_texts = [big], [""]
        
        unit_count = len(q_texts)
        # print(f"[QID={qid}] 找到 {unit_count} 个子查询+描述 单元")

        safe_qid = str(qid).replace('/', '_')
        if embed_method == "joint":
            units = [f"{qt} {dt}".strip() for qt, dt in zip(q_texts, d_texts)]
            unit_embs = model.encode(
                units,
                show_progress_bar=False,
                batch_size=batch_size,
                normalize_embeddings=True
            )
            np.save(os.path.join(joint_emb_dir, f"{safe_qid}.npy"), unit_embs)

        else:
            q_embs = model.encode(
                q_texts,
                show_progress_bar=False,
                batch_size=batch_size,
                normalize_embeddings=True
            )
            d_embs = model.encode(
                d_texts,
                show_progress_bar=False,
                batch_size=batch_size,
                normalize_embeddings=True
            )
            np.save(os.path.join(q_emb_dir, f"{safe_qid}.npy"), q_embs)
            np.save(os.path.join(d_emb_dir, f"{safe_qid}.npy"), d_embs)

            unit_embs = desc_weight * d_embs + (1 - desc_weight) * q_embs

        if fusion_method == "sum":
            fusion_buf = np.zeros(len(doc_ids), dtype=float)
        else:
            fusion_buf = np.full(len(doc_ids), -np.inf, dtype=float)

        per_subq_hits[qid] = {}
        per_subq_docs[qid] = {}

        for idx, uemb in enumerate(unit_embs, start=1):
            sims = cosine_similarity(uemb.reshape(1, -1), doc_emb).flatten()
            topk_idx = np.argsort(-sims)[:1000]
            topk_ids = [doc_ids[i] for i in topk_idx]

            per_subq_hits[qid][f"Unit{idx}"] = (
                len(set(topk_ids) & ground_truth.get(str(qid), set()))
                if ground_truth else 0
            )
            per_subq_docs[qid][f"Unit{idx}"] = topk_ids

            if fusion_method == "sum":
                fusion_buf += sims
            else:
                fusion_buf = np.maximum(fusion_buf, sims)

        for did in excluded_ids.get(str(qid), []):
            if did in doc_ids:
                fusion_buf[doc_ids.index(did)] = -np.inf

        final_idx = np.argsort(-fusion_buf)[:1000]
        fused_scores[str(qid)] = {
            doc_ids[i]: float(fusion_buf[i]) for i in final_idx
        }
        if ground_truth:
            fused_hit_counts[str(qid)] = len(
                set(fused_scores[str(qid)].keys()) &
                ground_truth[str(qid)]
            )
        else:
            fused_hit_counts[str(qid)] = 0

    return fused_scores, per_subq_hits, per_subq_docs, fused_hit_counts


@torch.no_grad()
def retrieval_bm25(queries,query_ids,documents,doc_ids,excluded_ids,long_context,**kwargs):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    all_scores = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query in zip(query_ids, queries):
        bar.update(1)
        query = analyzer.analyze(query)
        bm25_query = model[dictionary.doc2bow(query)]
        similarities = bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        for did, s in zip(doc_ids, similarities):
            all_scores[str(query_id)][did] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                all_scores[str(query_id)].pop(did)
        cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)[:1000]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]
    return all_scores

@torch.no_grad()
# sub-query+desc-version
def retrieval_bm25_fusion_desc(
    queries, query_ids, documents, doc_ids, excluded_ids, long_context,
    ground_truth=None,
    **kwargs
):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity

    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[[dictionary.doc2bow(doc) for doc in corpus]]
    bm25_index = SparseMatrixSimilarity(
        bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
        normalize_queries=False, normalize_documents=False
    )

    fused_scores     = {}
    per_subq_hits    = {}
    per_subq_docs    = {}
    fused_hit_counts = {}

    unit_pattern = re.compile(
        r'(Sub_Query_\d+:\s*"<begin_of_query>.*?<end_of_query>")\s*'
        r'(Desc\d+:\s*"<begin_of_desc>.*?<end_of_desc>")',
        flags=re.DOTALL
    )


    for qid, expanded_query in tqdm(list(zip(query_ids, queries)), desc="BM25 fusion_desc", total=len(queries)):
        units = []
        for m in unit_pattern.finditer(expanded_query):
            query_text = m.group(1).strip()
            desc_text = m.group(2).strip()
            units.append(f"{query_text} {desc_text}")

        if not units:
            units = [expanded_query.strip()]

        per_subq_hits[qid] = {}
        per_subq_docs[qid] = {}

        # print(f"[DEBUG] qid={qid} has {len(units)} units")

        for idx, unit in enumerate(units, start=1):
            tokens = analyzer.analyze(unit)
            bm25_query = model[dictionary.doc2bow(tokens)]
            sims = bm25_index[bm25_query].tolist()

            did_score_pairs = sorted(zip(doc_ids, sims), key=lambda x: x[1], reverse=True)[:1000]
            topk_docs = [did for did, _ in did_score_pairs]

            if ground_truth is not None and str(qid) in ground_truth:
                gold_set = ground_truth[str(qid)]
                hit_cnt = len(set(topk_docs) & gold_set)
            else:
                hit_cnt = 0

            per_subq_hits[qid][f"Unit{idx}"] = hit_cnt
            per_subq_docs[qid][f"Unit{idx}"] = topk_docs

        fusion_scores_this_q = {did: 0.0 for did in doc_ids}
        for unit in units:
            tokens = analyzer.analyze(unit)
            bm25_query = model[dictionary.doc2bow(tokens)]
            sims = bm25_index[bm25_query].tolist()
            for did, score in zip(doc_ids, sims):
                fusion_scores_this_q[did] += score

        for did in set(excluded_ids.get(str(qid), [])):
            fusion_scores_this_q.pop(did, None)

        topk = sorted(fusion_scores_this_q.items(), key=lambda x: x[1], reverse=True)[:1000]
        fused_scores[str(qid)] = {did: sc for did, sc in topk}

        topk_docs_fused = set(fused_scores[str(qid)].keys())
        if ground_truth is not None and str(qid) in ground_truth:
            gold_set = ground_truth[str(qid)]
            fused_hit_counts[str(qid)] = len(topk_docs_fused & gold_set)
        else:
            fused_hit_counts[str(qid)] = 0

    return fused_scores, per_subq_hits, per_subq_docs, fused_hit_counts


@torch.no_grad()
def retrieval_instructor(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='inst-l':
        model = SentenceTransformer('hkunlp/instructor-large')
    elif model_id=='inst-xl':
        model = SentenceTransformer('hkunlp/instructor-xl')
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model.set_pooling_include_prompt(False)

    batch_size = kwargs.get('batch_size',4)
    model.max_seq_length = kwargs.get('doc_max_length',2048)
    # queries = add_instruct_list(texts=queries,task=task,instruction=instructions['query'])
    # documents = add_instruct_list(texts=documents,task=task,instruction=instructions['document'])

    query_embs = model.encode(queries,batch_size=batch_size,show_progress_bar=True,prompt=instructions['query'].format(task=task),normalize_embeddings=True)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_embs = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_embs = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True,prompt=instructions['document'].format(task=task))
        np.save(cur_cache_file, doc_embs)
    scores = cosine_similarity(query_embs, doc_embs)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_grit(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    customized_checkpoint = kwargs.get('checkpoint',None)
    if customized_checkpoint is None:
        customized_checkpoint = 'GritLM/GritLM-7B'
    else:
        print('use',customized_checkpoint)
    model = GritLM(customized_checkpoint, torch_dtype="auto", mode="embedding")
    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    query_max_length = kwargs.get('query_max_length',256)
    doc_max_length = kwargs.get('doc_max_length',2048)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    ignore_cache = kwargs.pop('ignore_cache',False)
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file, allow_pickle=True)
    elif ignore_cache:
        doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
    else:
        doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
        np.save(cur_cache_file, doc_emb)
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=1, max_length=query_max_length)
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_openai(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    new_queries = []
    for q in queries:
        new_queries.append(cut_text_openai(text=q,tokenizer=tokenizer))
    queries = new_queries
    new_documents = []
    for d in documents:
        new_documents.append(cut_text_openai(text=d,tokenizer=tokenizer))
    documents = new_documents
    doc_emb = []
    batch_size = kwargs.get('batch_size',1024)
    # openai_client = OpenAI(api_key=kwargs['key'])
    openai_client = OpenAI()
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            cur_emb = get_embedding_openai(texts=documents[idx:idx + batch_size],openai_client=openai_client,tokenizer=tokenizer)
            with open(cur_cache_file,'w') as f:
                json.dump(cur_emb,f,indent=2)
        doc_emb += cur_emb
    query_emb = []
    for idx in trange(0, len(queries), batch_size):
        cur_emb = get_embedding_openai(texts=queries[idx:idx + batch_size], openai_client=openai_client,
                                       tokenizer=tokenizer)
        query_emb += cur_emb
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_cohere(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    query_emb = []
    doc_emb = []
    batch_size = kwargs.get('batch_size',8192)
    # cohere_client = cohere.Client(kwargs['key'])
    cohere_client = cohere.Client()
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            success = False
            exec_count = 0
            cur_emb = []
            while not success:
                exec_count += 1
                if exec_count>5:
                    print('cohere execute too many times')
                    exit(0)
                try:
                    cur_emb = cohere_client.embed(texts=documents[idx:idx+batch_size], input_type="search_document",
                                                  model="embed-english-v3.0").embeddings

                    success = True
                except Exception as e:
                    print(e)
                    time.sleep(60)
            with open(cur_cache_file, 'w') as f:
                json.dump(cur_emb, f, indent=2)
        doc_emb += cur_emb
    for idx in trange(0, len(queries), batch_size):
        success = False
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('cohere query execute too many times')
                exit(0)
            try:
                cur_emb = cohere_client.embed(queries[idx:idx+batch_size], input_type="search_query",
                                              model="embed-english-v3.0").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                time.sleep(60)
    scores = (torch.tensor(query_emb) @ torch.tensor(doc_emb).T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_voyage(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
    new_queries = []
    for q in queries:
        new_queries.append(cut_text(text=q,tokenizer=tokenizer,threshold=16000))
    queries = new_queries
    new_documents = []
    for d in tqdm(documents,desc='preprocess documents'):
        new_documents.append(cut_text(text=d,tokenizer=tokenizer,threshold=16000))
    documents = new_documents

    query_emb = []
    doc_emb = []

    batch_size = kwargs.get('batch_size',1)

    doc_emb = None
    doc_cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(doc_cache_path), exist_ok=True)
    if os.path.isfile(doc_cache_path):
        # already exists so we can just load it
        doc_emb = np.load(doc_cache_path, allow_pickle=True)

    # voyage_client = voyageai.Client(api_key=kwargs['key'])
    voyage_client = voyageai.Client()
    for i in trange(0,len(documents),batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > i:
            continue
        
        success = False
        threshold = 16000
        cur_texts = documents[i:i+batch_size]
        count_over = 0
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('voyage document too many times')
                exit(0)
            try:
                cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="document").embeddings
                doc_emb = cur_emb if doc_emb is None else np.concatenate((doc_emb, np.array(cur_emb)), axis=0)
                if (i + 1) % 1000 == 0:
                    np.save(doc_cache_path, doc_emb)
                success = True
            except Exception as e:
                print(e)
                count_over += 1
                threshold = threshold-500
                if count_over>4:
                    print('voyage:',count_over)
                new_texts = []
                for t in cur_texts:
                    new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                cur_texts = new_texts
                time.sleep(5)

    query_emb = []
    for i in trange(0,len(queries),batch_size):
        success = False
        threshold = 16000
        cur_texts = queries[i:i+batch_size]
        count_over = 0
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('voyage query execute too many times')
                exit(0)
            try:
                cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="query").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                count_over += 1
                threshold = threshold-500
                if count_over>4:
                    print('voyage:',count_over)
                new_texts = []
                for t in cur_texts:
                    new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                cur_texts = new_texts
                time.sleep(60)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_google(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    model = TextEmbeddingModel.from_pretrained("text-embedding-preview-0409")
    query_emb = []
    # doc_emb = []
    batch_size = kwargs.get('batch_size',8)
    doc_emb = None
    cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        # already exists so we can just load it
        doc_emb = np.load(cache_path, allow_pickle=True)

    for start_idx in tqdm(range(0, len(documents), batch_size), desc='embedding'):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue
        
        cur_emb = get_embedding_google(
            texts=documents[start_idx:start_idx + batch_size], task='RETRIEVAL_DOCUMENT',
            model=model
        )
        doc_emb = cur_emb if doc_emb is None else np.concatenate((doc_emb, np.array(cur_emb)), axis=0)
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)
    np.save(cache_path, doc_emb)
        
    for start_idx in tqdm(range(0,len(queries), batch_size),desc='embedding'):
        query_emb += get_embedding_google(texts=queries[start_idx:start_idx+ batch_size],task='RETRIEVAL_QUERY',model=model)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


RETRIEVAL_FUNCS = {
    'sf': retrieval_sf_qwen_e5,
    'qwen': retrieval_sf_qwen_e5,
    'qwen2': retrieval_sf_qwen_e5,
    'e5': retrieval_sf_qwen_e5,
    'bm25': retrieval_bm25,
    'bm25_fusion_desc': retrieval_bm25_fusion_desc,
    'sbert': retrieval_sbert_bge,
    'bge': retrieval_sbert_bge,
    'sbert_fusion_desc': retrieval_sbert_bge_fusion_desc,
    'bge_fusion_desc': retrieval_sbert_bge_fusion_desc,
    'inst-l': retrieval_instructor,
    'inst-xl': retrieval_instructor,
    'grit': retrieval_grit,
    'cohere': retrieval_cohere,
    'voyage': retrieval_voyage,
    'openai': retrieval_openai,
    'google': retrieval_google
}

def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    print(output)
    return output
