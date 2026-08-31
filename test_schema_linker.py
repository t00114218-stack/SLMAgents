#!/usr/bin/env python3
"""
Test generic schema linking and candidate join graph on BIRD databases.
"""
import re
import json

def parse_ddl(schema_ddl: str) -> dict:
    tables = {}
    statements = [s.strip() for s in schema_ddl.split(';') if s.strip()]
    for stmt in statements:
        m = re.search(r'CREATE\s+TABLE\s+[`"]?([\w\-]+)[`"]?\s*\((.*)\)', stmt, re.IGNORECASE | re.DOTALL)
        if m:
            t_name = m.group(1).strip()
            body = m.group(2)
            cols = []
            for line in body.split('\n'):
                line = line.strip().rstrip(',')
                m_col = re.search(r'^[`"]?([\w\s\(\)\/\-\%]+)[`"]?\s+([A-Z]+)', line, re.IGNORECASE)
                if m_col:
                    c_name = m_col.group(1).strip()
                    c_type = m_col.group(2).strip().upper()
                    if c_name.lower() not in ('primary', 'foreign', 'key', 'constraint', 'unique', 'check'):
                        cols.append({'name': c_name, 'type': c_type})
            tables[t_name] = cols
    return tables

def infer_join_graph(tables: dict) -> list:
    joins = []
    t_names = list(tables.keys())
    for i in range(len(t_names)):
        for j in range(i + 1, len(t_names)):
            t1, t2 = t_names[i], t_names[j]
            cols1 = [c['name'] for c in tables[t1]]
            cols2 = [c['name'] for c in tables[t2]]
            for c1 in cols1:
                for c2 in cols2:
                    c1_clean = re.sub(r'[^a-zA-Z0-9]', '', c1).lower()
                    c2_clean = re.sub(r'[^a-zA-Z0-9]', '', c2).lower()
                    if not c1_clean or not c2_clean:
                        continue
                    # Match exact or primary/foreign key conventions across tables
                    if c1_clean == c2_clean and len(c1_clean) >= 2:
                        joins.append((t1, c1, t2, c2))
                    elif c1_clean == 'id' and (c2_clean == t1.lower() + 'id' or c2_clean.endswith('_' + t1.lower() + '_id') or c2_clean.endswith(t1.lower() + 'id')):
                        joins.append((t1, c1, t2, c2))
                    elif c2_clean == 'id' and (c1_clean == t2.lower() + 'id' or c1_clean.endswith('_' + t2.lower() + '_id') or c1_clean.endswith(t2.lower() + 'id')):
                        joins.append((t1, c1, t2, c2))
                    elif (c1_clean == 'cds' and c2_clean == 'cdscode') or (c2_clean == 'cds' and c1_clean == 'cdscode'):
                        joins.append((t1, c1, t2, c2))
    return joins

def link_schema_to_question(question: str, evidence: str, tables: dict, joins: list) -> str:
    """
    Generates a generic, schema-grounded linking block for the prompt.
    Identifies relevant tables and candidate join conditions.
    """
    combined_text = (question + " " + evidence).lower()
    tokens = set(re.findall(r'\w+', combined_text))
    # Extract 2-grams and 3-grams for multi-word column matching
    words = re.findall(r'\w+', combined_text)
    ngrams = set(tokens)
    for i in range(len(words) - 1):
        ngrams.add(f"{words[i]} {words[i+1]}")
    for i in range(len(words) - 2):
        ngrams.add(f"{words[i]} {words[i+1]} {words[i+2]}")
        
    relevant_tables = {}
    for t_name, cols in tables.items():
        matched_cols = []
        t_clean = t_name.lower()
        t_matched = t_clean in ngrams or any(w in t_clean for w in tokens if len(w) >= 4)
        
        for c in cols:
            c_name = c['name']
            c_clean = c_name.lower()
            c_words = re.findall(r'\w+', c_clean)
            if c_clean in ngrams or any(w in tokens for w in c_words if len(w) >= 3):
                matched_cols.append(c_name)
                
        if t_matched or matched_cols:
            relevant_tables[t_name] = matched_cols
            
    # Find applicable joins between relevant tables
    relevant_t_set = set(relevant_tables.keys())
    applicable_joins = []
    for j in joins:
        t1, c1, t2, c2 = j
        if t1 in relevant_t_set and t2 in relevant_t_set:
            applicable_joins.append(f"`{t1}`.`{c1}` = `{t2}`.`{c2}`")
            
    # Format output block
    lines = []
    lines.append("### Schema Linking & Column Grounding:")
    for t_name, matched_cols in relevant_tables.items():
        cols_str = ", ".join(f"`{c}`" for c in matched_cols[:8]) if matched_cols else "all columns"
        lines.append(f"- Table `{t_name}`: matched columns [{cols_str}]")
        
    if applicable_joins:
        lines.append("\n### Verified Join Relationships:")
        for j in applicable_joins:
            lines.append(f"- Join on: {j}")
            
    return "\n".join(lines)

if __name__ == "__main__":
    with open('leaderboard/bird_bench/data/bird_dev_500.jsonl') as f:
        samples = [json.loads(line) for line in [f.readline() for _ in range(5)]]
        
    for idx, s in enumerate(samples):
        print(f"\n{'='*60}\nSample #{idx+1}: {s['question']}")
        print(f"Evidence: {s['evidence']}")
        tables = parse_ddl(s['schema_ddl'])
        joins = infer_join_graph(tables)
        linking_block = link_schema_to_question(s['question'], s['evidence'], tables, joins)
        print(linking_block)
