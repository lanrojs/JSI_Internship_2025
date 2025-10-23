#!/usr/bin/env python3
"""
Build a GraphRAG-style knowledge graph from Agentic Contextual Ingestion outputs.

Inputs (from <doc>_contextualized/):
  - passages.jsonl          # one per passage
  - entities.jsonl          # entity mentions per passage
  - relationships.jsonl     # typed inter-entity edges with evidence (passage_id)
  - claims.jsonl            # claims keyed by (doc_id, passage_id, text)
  - vec_index.npy           # optional, passage embeddings (contextualized)
  - vec_meta.jsonl          # optional, aligns vectors to passage_ids

Outputs:
  - graph.json              # {"nodes":[...], "edges":[...]} with stable ids
  - (optional) Neo4j CSVs   # nodes_*.csv, edges_*.csv
  - (optional) GraphML      # graph.graphml

Usage:
  python build_graphrag.py \
      --input-dir /path/to/<doc>_contextualized \
      --out graph.json \
      --neo4j-out /path/to/neo4j_csv \
      --graphml-out graph.graphml \
      --sim-topk 3 \
      --sim-threshold 0.35
"""

from __future__ import annotations
import os, re, json, argparse, math, csv, sys, hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional, Set


try:
    import orjson
    HAVE_ORJSON = True
except Exception:
    HAVE_ORJSON = False

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

try:
    import networkx as nx
    HAVE_NETWORKX = True
except Exception:
    HAVE_NETWORKX = False


# -----------------------------
# Utilities
# -----------------------------
def load_jsonl(path: str) -> List[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(orjson.loads(line) if HAVE_ORJSON else json.loads(line))
            except Exception:
                # best-effort: skip bad lines
                continue
    return out

def stable_id(*parts: str, n: int = 32) -> str:
    return hashlib.sha256(("||".join(parts)).encode("utf-8")).hexdigest()[:n]

def jdump(obj: Any) -> str:
    if HAVE_ORJSON:
        return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode()
    return json.dumps(obj, indent=2, ensure_ascii=False)

def cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    num = float((a * b).sum())
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return num / den


# -----------------------------
# Graph schema
# -----------------------------
@dataclass
class Node:
    id: str
    label: str  # "Entity" | "Passage" | "Claim"
    # Common props:
    doc_id: str = ""
    name: str = ""         # Entity or Claim text (short)
    type: str = ""         # Entity subtype (role/actor/concept/...) or blank
    text: str = ""         # Full text (for Passage/Claim)
    idx: int = -1          # Passage index if label == "Passage"
    canonical_id: str = "" # For Entity nodes

@dataclass
class Edge:
    id: str
    src: str
    dst: str
    label: str  # "RELATES" | "MENTIONS" | "SUPPORTS" | "SIMILAR_TO"
    # Common props:
    evidence: str = ""     # evidence span or reason
    passage_id: str = ""   # where extracted from if applicable
    rel_type: str = ""     # for RELATES edges (obligation, scope, ...)
    score: float = 0.0     # for similarity edges

class GraphStore:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}

    def upsert_node(self, n: Node):
        self.nodes[n.id] = n

    def add_edge(self, e: Edge):
        self.edges[e.id] = e

    def to_json(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
        }


# -----------------------------
# Builders
# -----------------------------
def build_graph(
    input_dir: str,
    sim_topk: int = 3,
    sim_threshold: float = 0.35,
) -> GraphStore:
    g = GraphStore()

    # Load inputs
    passages = load_jsonl(os.path.join(input_dir, "passages.jsonl"))           # fields: id, doc_id, idx, text, contextualized, ...
    entities  = load_jsonl(os.path.join(input_dir, "entities.jsonl"))          # fields: passage_id, name, type, canonical_id, doc_id
    relationships = load_jsonl(os.path.join(input_dir, "relationships.jsonl")) # fields: source, target, type, evidence, passage_id, source_id/target_id, doc_id
    claims    = load_jsonl(os.path.join(input_dir, "claims.jsonl"))            # fields: doc_id, passage_id, claim

    # 1) Passage nodes
    pid_to_docidx: Dict[str, Tuple[str, int]] = {}
    for p in passages:
        pid = p.get("id") or stable_id("passage", p.get("doc_id",""), str(p.get("idx","")), p.get("text","")[:64])
        doc_id = p.get("doc_id","")
        idx = int(p.get("idx", -1))
        text = str(p.get("contextualized") or p.get("text") or "")
        node = Node(
            id=pid, label="Passage", doc_id=doc_id, text=text, idx=idx, name=f"{doc_id}#p{idx}"
        )
        g.upsert_node(node)
        pid_to_docidx[pid] = (doc_id, idx)

    # 2) Entity nodes + MENTIONS edges (passage -> entity)
    # Deduplicate entities by canonical_id if present, else by (doc_id, name)
    ent_key_to_id: Dict[Tuple[str, str], str] = {}
    passage_to_entities: Dict[str, Set[str]] = {}

    for e in entities:
        doc_id = e.get("doc_id","")
        passage_id = e.get("passage_id","")
        name = (e.get("name") or "").strip()
        etype = e.get("type","")
        canonical = (e.get("canonical_id") or "").strip()

        if not name:
            continue

        key = ("cid", canonical) if canonical else ("name", f"{doc_id}::{name.lower()}")
        if key not in ent_key_to_id:
            eid = canonical if canonical else stable_id("ent", doc_id, name.lower())
            ent_key_to_id[key] = eid
            node = Node(
                id=eid,
                label="Entity",
                doc_id=doc_id,
                name=name,
                type=etype,
                canonical_id=canonical or eid
            )
            g.upsert_node(node)
        eid = ent_key_to_id[key]

        if passage_id:
            # track mention set
            passage_to_entities.setdefault(passage_id, set()).add(eid)

    # Add MENTIONS edges after we know deduped entity ids
    for passage_id, ent_ids in passage_to_entities.items():
        for eid in ent_ids:
            eid_edge = stable_id("MENTIONS", passage_id, eid)
            g.add_edge(Edge(id=eid_edge, src=passage_id, dst=eid, label="MENTIONS"))

    # 3) Claim nodes + SUPPORTS edges (passage -> claim)
    # Deduplicate claims by (doc_id, normalized text)
    claim_key_to_id: Dict[Tuple[str, str], str] = {}
    for c in claims:
        doc_id = c.get("doc_id","")
        passage_id = c.get("passage_id","")
        text = (c.get("claim") or "").strip()
        if not text:
            continue
        key = (doc_id, text.lower())
        if key not in claim_key_to_id:
            cid = stable_id("claim", doc_id, text)
            claim_key_to_id[key] = cid
            node = Node(id=cid, label="Claim", doc_id=doc_id, name=text[:80], text=text)
            g.upsert_node(node)
        cid = claim_key_to_id[key]
        if passage_id:
            eid_edge = stable_id("SUPPORTS", passage_id, cid)
            g.add_edge(Edge(id=eid_edge, src=passage_id, dst=cid, label="SUPPORTS"))

    # 4) RELATES edges (entity -> entity) from relationships.jsonl
    # Prefer canonical_ids if present; otherwise resolve via name to our entity map.
    name_index: Dict[str, str] = {}  # name (lower) -> eid (best-effort)
    for n in g.nodes.values():
        if n.label == "Entity" and n.name:
            name_index.setdefault((n.doc_id + "::" + n.name.lower()), n.id)

    for r in relationships:
        doc_id = r.get("doc_id","")
        src_id = (r.get("source_id") or "").strip()
        dst_id = (r.get("target_id") or "").strip()
        src_name = (r.get("source") or "").strip()
        dst_name = (r.get("target") or "").strip()
        rel_type = r.get("type","")
        evidence = r.get("evidence","")
        passage_id = r.get("passage_id","")

        # Resolve missing IDs via name + doc scope:
        if not src_id and src_name:
            src_id = name_index.get(doc_id + "::" + src_name.lower(), "")
        if not dst_id and dst_name:
            dst_id = name_index.get(doc_id + "::" + dst_name.lower(), "")

        if not src_id or not dst_id or src_id == dst_id:
            continue

        eid = stable_id("RELATES", src_id, dst_id, rel_type or "", passage_id or "")
        g.add_edge(Edge(
            id=eid, src=src_id, dst=dst_id, label="RELATES",
            rel_type=rel_type, evidence=evidence, passage_id=passage_id
        ))

    # 5) SIMILAR_TO (passage ↔ passage) based on vec_index.npy + vec_meta.jsonl
    vec_npy = os.path.join(input_dir, "vec_index.npy")
    vec_meta = os.path.join(input_dir, "vec_meta.jsonl")
    if HAVE_NUMPY and os.path.exists(vec_npy) and os.path.exists(vec_meta) and sim_topk > 0:
        try:
            vecs = np.load(vec_npy)
            metas = load_jsonl(vec_meta)  # [{"id": <passage_id>, "text": ...}, ...] in your pipeline
            # Align meta ids with vec rows
            meta_ids = [m.get("id","") for m in metas]
            id_to_row = {pid: i for i, pid in enumerate(meta_ids) if pid}
            # Collect passage ids present both in graph and vectors
            pids = [n.id for n in g.nodes.values() if n.label == "Passage" and n.id in id_to_row]
            for pid in pids:
                i = id_to_row[pid]
                v = vecs[i].astype("float32")
                # cosine to all others (top-k excluding self)
                sims: List[Tuple[float, str]] = []
                for pid2 in pids:
                    if pid2 == pid:
                        continue
                    j = id_to_row[pid2]
                    s = cosine(v, vecs[j].astype("float32"))
                    sims.append((s, pid2))
                sims.sort(reverse=True, key=lambda x: x[0])
                for s, pid2 in sims[:sim_topk]:
                    if s < sim_threshold:
                        continue
                    # undirected de-dup: only add if pid < pid2
                    a, b = sorted([pid, pid2])
                    eid = stable_id("SIM", a, b)
                    if eid not in g.edges:
                        g.add_edge(Edge(id=eid, src=a, dst=b, label="SIMILAR_TO", score=float(s)))
        except Exception as ex:
            print(f"[warn] similarity edges skipped: {ex}", file=sys.stderr)

    return g


# -----------------------------
# Exporters
# -----------------------------
def export_json(g: GraphStore, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(jdump(g.to_json()))

def export_neo4j_csv(g: GraphStore, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Nodes by label
    node_files = {
        "Entity": open(os.path.join(out_dir, "nodes_Entity.csv"), "w", newline="", encoding="utf-8"),
        "Passage": open(os.path.join(out_dir, "nodes_Passage.csv"), "w", newline="", encoding="utf-8"),
        "Claim": open(os.path.join(out_dir, "nodes_Claim.csv"), "w", newline="", encoding="utf-8"),
    }
    edge_files = {
        "RELATES": open(os.path.join(out_dir, "edges_RELATES.csv"), "w", newline="", encoding="utf-8"),
        "MENTIONS": open(os.path.join(out_dir, "edges_MENTIONS.csv"), "w", newline="", encoding="utf-8"),
        "SUPPORTS": open(os.path.join(out_dir, "edges_SUPPORTS.csv"), "w", newline="", encoding="utf-8"),
        "SIMILAR_TO": open(os.path.join(out_dir, "edges_SIMILAR_TO.csv"), "w", newline="", encoding="utf-8"),
    }

    node_writers = {
        k: csv.DictWriter(v, fieldnames=[
            "id:ID", ":LABEL", "doc_id", "name", "type", "text", "idx:int", "canonical_id"
        ])
        for k, v in node_files.items()
    }
    for w in node_writers.values():
        w.writeheader()

    for n in g.nodes.values():
        node_writers[n.label].writerow({
            "id:ID": n.id,
            ":LABEL": n.label,
            "doc_id": n.doc_id,
            "name": n.name,
            "type": n.type,
            "text": n.text,
            "idx:int": n.idx if n.idx >= 0 else "",
            "canonical_id": n.canonical_id,
        })

    edge_writers = {
        k: csv.DictWriter(v, fieldnames=[
            ":START_ID", ":END_ID", ":TYPE", "rel_type", "evidence", "passage_id", "score:float", "id"
        ])
        for k, v in edge_files.items()
    }
    for w in edge_writers.values():
        w.writeheader()

    for e in g.edges.values():
        edge_writers[e.label].writerow({
            ":START_ID": e.src,
            ":END_ID": e.dst,
            ":TYPE": e.label,
            "rel_type": e.rel_type,
            "evidence": e.evidence,
            "passage_id": e.passage_id,
            "score:float": f"{e.score:.6f}" if e.score else "",
            "id": e.id,
        })

    # Close files
    for f in node_files.values():
        f.close()
    for f in edge_files.values():
        f.close()

    # Helper import command (printed)
    print("\nNeo4j bulk import example:")
    print(f"  neo4j-admin database import full --overwrite-destination true --nodes={os.path.join(out_dir, 'nodes_Entity.csv')} "
          f"--nodes={os.path.join(out_dir, 'nodes_Passage.csv')} --nodes={os.path.join(out_dir, 'nodes_Claim.csv')} "
          f"--relationships={os.path.join(out_dir, 'edges_RELATES.csv')} "
          f"--relationships={os.path.join(out_dir, 'edges_MENTIONS.csv')} "
          f"--relationships={os.path.join(out_dir, 'edges_SUPPORTS.csv')} "
          f"--relationships={os.path.join(out_dir, 'edges_SIMILAR_TO.csv')} "
          f"<your-db-name>")

def export_graphml(g: GraphStore, out_path: str):
    if not HAVE_NETWORKX:
        print("[warn] networkx not installed; skipping GraphML", file=sys.stderr)
        return
    G = nx.Graph()
    # Use a MultiDiGraph for full fidelity? GraphML in NX has limited multi-edge support;
    # we’ll encode edges uniquely and keep properties.
    for n in g.nodes.values():
        G.add_node(n.id, label=n.label, doc_id=n.doc_id, name=n.name, type=n.type,
                   text=n.text, idx=n.idx, canonical_id=n.canonical_id)
    for e in g.edges.values():
        # For SIMILAR_TO we add undirected; others we can still place as undirected for GraphML simplicity
        G.add_edge(e.src, e.dst, key=e.id, type=e.label, rel_type=e.rel_type,
                   evidence=e.evidence, passage_id=e.passage_id, score=e.score, id=e.id)
    nx.write_graphml(G, out_path)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build GraphRAG-style graph from contextualized outputs.")
    ap.add_argument("--input-dir", required=True, help="Path to <doc>_contextualized directory")
    ap.add_argument("--out", default="graph.json", help="Output JSON path")
    ap.add_argument("--neo4j-out", default="", help="If set, write Neo4j CSVs to this directory")
    ap.add_argument("--graphml-out", default="", help="If set, write GraphML here (requires networkx)")
    ap.add_argument("--sim-topk", type=int, default=3, help="Top-k passage neighbors for SIMILAR_TO")
    ap.add_argument("--sim-threshold", type=float, default=0.35, help="Cosine threshold for SIMILAR_TO")
    args = ap.parse_args()

    g = build_graph(
        input_dir=args.input_dir,
        sim_topk=max(0, args.sim_topk),
        sim_threshold=float(args.sim_threshold),
    )

    export_json(g, args.out)
    print(f"[ok] wrote {args.out} with {len(g.nodes)} nodes and {len(g.edges)} edges")

    if args.neo4j_out:
        export_neo4j_csv(g, args.neo4j_out)
        print(f"[ok] wrote Neo4j CSVs to {args.neo4j_out}")

    if args.graphml_out:
        export_graphml(g, args.graphml_out)
        print(f"[ok] wrote GraphML to {args.graphml_out}")

if __name__ == "__main__":
    main()
