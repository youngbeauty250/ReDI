import os
import argparse
import json
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS,calculate_retrieval_metrics
from datasets import Dataset, load_dataset

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics',
                                 'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
                                 'theoremqa_questions'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25','bm25_fusion_desc','cohere','e5','google','grit','inst-l','inst-xl',
                                 'openai','qwen','qwen2','sbert','sbert_fusion_desc','sf','voyage','bge'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--query_max_length', type=int, default=-1)
    parser.add_argument('--doc_max_length', type=int, default=-1)
    parser.add_argument('--encode_batch_size', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    args = parser.parse_args()
    if args.model == "bm25_fusion_desc":
        args.output_dir = os.path.join(args.output_dir,f"{args.task}_bm25_long_{args.long_context}")
    elif args.model == "sbert_fusion_desc":
        args.output_dir = os.path.join(args.output_dir,f"{args.task}_dense_long_{args.long_context}")
    else:
        args.output_dir = os.path.join(args.output_dir,f"{args.task}_{args.model}_long_{args.long_context}")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    score_file_path = os.path.join(args.output_dir,f'score.json')

    if args.input_file is not None:
        with open(args.input_file) as f:
            examples = json.load(f)
    elif args.reasoning is not None:
        # examples = load_dataset('xlangai/bright', f"{args.reasoning}_reason", cache_dir=args.cache_dir)[args.task]
        # Load from local path, replace with your path
        examples = Dataset.from_file(
            f"/.../{args.reasoning}_reason/bright-{args.task}.arrow"
        )
    else:
        examples = load_dataset('xlangai/bright', 'examples',cache_dir=args.cache_dir)[args.task]
    if args.long_context:
        doc_pairs = load_dataset('xlangai/bright', 'long_documents',cache_dir=args.cache_dir)[args.task]
    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents',cache_dir=args.cache_dir)[args.task]
    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(dp['id'])
        documents.append(dp['content'])

    if not os.path.isfile(score_file_path):
        if args.model in ("bm25_fusion_desc"):
            with open(os.path.join(args.config_dir,"bm25",f"{args.task}.json")) as f:
                config = json.load(f)
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
        elif args.model in ("sbert_fusion_desc"):
            with open(os.path.join(args.config_dir,"dense",f"{args.task}.json")) as f:
                config = json.load(f)
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
        else:
            with open(os.path.join(args.config_dir,args.model,f"{args.task}.json")) as f:
                config = json.load(f)
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)
         
        queries = []
        query_ids = []
        excluded_ids = {}
        for e in examples:
            queries.append(e["query"])
            query_ids.append(e['id'])
            excluded_ids[e['id']] = e['excluded_ids']
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
            assert len(overlap)==0
        assert len(queries)==len(query_ids), f"{len(queries)}, {len(query_ids)}"
        if not os.path.isdir(os.path.join(args.cache_dir, 'doc_ids')):
            os.makedirs(os.path.join(args.cache_dir, 'doc_ids'))
        if os.path.isfile(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json")):
            with open(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json")) as f:
                cached_doc_ids = json.load(f)
            for id1,id2 in zip(cached_doc_ids,doc_ids):
                assert id1==id2
        else:
            with open(os.path.join(args.cache_dir,'doc_ids',f"{args.task}_{args.long_context}.json"),'w') as f:
                json.dump(doc_ids,f,indent=2)
        assert len(doc_ids)==len(documents), f"{len(doc_ids)}, {len(documents)}"

        print(f"{len(queries)} queries")
        print(f"{len(documents)} documents")
        if args.debug:
            documents = documents[:30]
            doc_paths = doc_ids[:30]
        kwargs = {}
        if args.query_max_length>0:
            kwargs = {'query_max_length': args.query_max_length}
        if args.doc_max_length>0:
            kwargs.update({'doc_max_length': args.doc_max_length})
        if args.encode_batch_size>0:
            kwargs.update({'batch_size': args.encode_batch_size})
        if args.key is not None:
            kwargs.update({'key': args.key})
        if args.ignore_cache:
            kwargs.update({'ignore_cache': args.ignore_cache})
            
        if args.model in ("bm25_fusion_desc","sbert_fusion_desc"):
            ground_truth = { str(e["id"]): set(e["gold_ids"]) for e in examples }
            fused_scores, per_subq_hits, per_subq_docs, fused_hit_counts = RETRIEVAL_FUNCS[args.model](
                queries=queries, query_ids=query_ids,
                documents=documents, doc_ids=doc_ids,
                excluded_ids=excluded_ids, long_context=args.long_context,
                instructions=config["instructions_long"] if args.long_context else config["instructions"],
                model_id=args.model,
                ground_truth=ground_truth,  
                checkpoint=args.checkpoint,
                key=args.key,
                ignore_cache=args.ignore_cache
            )
            scores = fused_scores
            
            for qid, hits in per_subq_hits.items():
                total = len(ground_truth[str(qid)])
                for unit, cnt in hits.items():
                    per_subq_hits[qid][unit] = f"{cnt}/{total}"
            with open(os.path.join(args.output_dir, "per_subq_hits.json"), "w") as f:
                json.dump(per_subq_hits, f, indent=2)
            with open(os.path.join(args.output_dir, "per_subq_docs.json"), "w") as f:
                json.dump(per_subq_docs, f, indent=2)
            oracle_stats = {}
            for qid, docs_dict in per_subq_docs.items():
                units = list(docs_dict.keys())
                total_docs = len(units) * 1000
                all_docs = []
                for lst in docs_dict.values():
                    all_docs.extend(lst)
                unique_docs = set(all_docs)
                total_docs_unique = len(unique_docs)
                gold_set = ground_truth[str(qid)]
                hit_gold = len(unique_docs & gold_set)
                total_gold = len(gold_set)
                unit_hits = {}
                for unit, lst in docs_dict.items():
                    unit_hits[unit] = len(set(lst) & gold_set)
                oracle_stats[qid] = {
                    "total_docs": total_docs,
                    "total_docs_unique": total_docs_unique,
                    "hit_gold_ids": hit_gold,
                    "total_gold_ids": total_gold,
                    "unit_hits": unit_hits
                }
            with open(os.path.join(args.output_dir, "oracle_stats.json"), "w") as f:
                json.dump(oracle_stats, f, indent=2)
        else:
            scores = RETRIEVAL_FUNCS[args.model](
                queries=queries, query_ids=query_ids, documents=documents, excluded_ids=excluded_ids,
                instructions=config['instructions_long'] if args.long_context else config['instructions'],
                doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir, long_context=args.long_context,
                model_id=args.model, checkpoint= args.checkpoint, **kwargs
            )
        with open(score_file_path,'w') as f:
            json.dump(scores,f,indent=2)
    else:
        with open(score_file_path) as f:
            scores = json.load(f)
        print(score_file_path,'exists')
    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for did in e['excluded_ids']:
            assert not did in scores[e['id']]
            assert not did in ground_truth[e['id']]
    
    hit_counts = {}
    for qid, retrieved_dict in scores.items():
        retrieved_set = set(retrieved_dict.keys())
        gold_set      = set(ground_truth[qid].keys())

        hit = len(retrieved_set & gold_set)
        total = len(gold_set)
        hit_counts[qid] = f"{hit}/{total}"
    
    if args.model == "bm25_fusion_desc":
        with open(os.path.join(args.output_dir, f"hit_counts_bm25.json"), "w") as hf:
            json.dump(hit_counts, hf, indent=2)
    elif args.model == "sbert_fusion_desc":
        with open(os.path.join(args.output_dir, f"hit_counts_dense.json"), "w") as hf:
            json.dump(hit_counts, hf, indent=2)
    else:
        with open(os.path.join(args.output_dir, f"hit_counts_{args.model}.json"), "w") as hf:
            json.dump(hit_counts, hf, indent=2)

    print(args.output_dir)
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
