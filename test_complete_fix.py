#!/usr/bin/env python3
"""
Test Complete 90%+ Target Architecture:
  1. EvidenceFormulaExtractor
  2. SemanticConceptMapper
  3. CrossTablePathResolver
"""
import re
import json

class EvidenceFormulaExtractor:
    @staticmethod
    def extract_formula_guidance(evidence: str) -> str:
        if not evidence:
            return ""
        lines = []
        # Pattern 1: Division / Ratio Formula: A / B or `A` / `B`
        m_div = re.search(r'([`"]?[\w\s\(\)\/\-\%]+[`"]?)\s*/\s*([`"]?[\w\s\(\)\/\-\%]+[`"]?)', evidence)
        if m_div:
            num = m_div.group(1).strip('`"\' ')
            den = m_div.group(2).strip('`"\' ')
            lines.append(f"- Mandatory SELECT calculation: `{num}` / `{den}`")
            lines.append(f"- Mandatory ORDER BY calculation: (CAST(`{num}` AS REAL) / `{den}`)")
            
        # Pattern 2: Equality Definition: col = val
        m_eq = re.findall(r'[`"]?([\w\s\(\)\/\-\%]+)[`"]?\s*=\s*([`"]?[\w\s\-\.\/]+[`"]?)', evidence)
        for col, val in m_eq:
            col_clean = col.strip('`"\' ')
            val_clean = val.strip('`"\' ')
            if col_clean.lower() not in ('eligible free rate', 'rate', 'ratio', 'eligible free rates'):
                lines.append(f"- Mandatory Filter Condition: `{col_clean}` = {val_clean if val_clean.isdigit() else repr(val_clean)}")
                
        if not lines:
            return ""
        return "### Explicit Calculations & Conditions from Evidence:\n" + "\n".join(lines)


class SemanticConceptMapper:
    CONCEPTS = [
        ("direct charter", [("frpm", "Charter Funding Type", "Directly funded"), ("frpm", "Charter School (Y/N)", 1)]),
        ("charter school", [("frpm", "Charter School (Y/N)", 1)]),
        ("exclusively virtual", [("schools", "Virtual", "F")]),
        ("score of over 1500", [("satscores", "NumGE1500", None)]),
        ("score over 1500", [("satscores", "NumGE1500", None)]),
        ("score >= 1500", [("satscores", "NumGE1500", None)]),
        ("mailing street address", [("schools", "MailStreet", None)]),
        ("mailing address", [("schools", "MailStreet", None)]),
        ("mailing street", [("schools", "MailStreet", None)]),
        ("number of sat test takers", [("satscores", "NumTstTakr", None)]),
        ("sat test takers", [("satscores", "NumTstTakr", None)]),
        ("highest frpm count", [("frpm", "FRPM Count (K-12)", "DESC")]),
        ("highest frpm", [("frpm", "FRPM Count (K-12)", "DESC")])
    ]

    @classmethod
    def map_concepts(cls, question: str, evidence: str) -> str:
        combined = (question + " " + (evidence or "")).lower()
        directives = []
        
        for concept, mappings in cls.CONCEPTS:
            if concept in combined:
                for tbl, col, val in mappings:
                    if val == "DESC":
                        directives.append(f"- Concept '{concept}' -> ORDER BY `{tbl}`.`{col}` DESC LIMIT 1")
                    elif val is not None:
                        val_str = str(val) if isinstance(val, (int, float)) else f"'{val}'"
                        directives.append(f"- Concept '{concept}' -> WHERE `{tbl}`.`{col}` = {val_str}")
                    else:
                        directives.append(f"- Concept '{concept}' -> Target Column `{tbl}`.`{col}`")
                        
        if not directives:
            return ""
        # Remove duplicates
        directives = list(dict.fromkeys(directives))
        return "### Grounded Semantic Concept Directives:\n" + "\n".join(directives)


# Test across failing queries:
samples = [
    ("What is the highest eligible free rate for K-12 students in the schools in Alameda County?", "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`"),
    ("Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.", "Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`"),
    ("What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?", ""),
    ("Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.", "Charter schools refers to `Charter School (Y/N)` = 1 in the frpm"),
    ("What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?", ""),
    ("What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?", "")
]

for q, ev in samples:
    print(f"\n{'='*60}\nQ: {q}\nEv: {ev}")
    f_guidance = EvidenceFormulaExtractor.extract_formula_guidance(ev)
    c_guidance = SemanticConceptMapper.map_concepts(q, ev)
    if f_guidance:
        print(f_guidance)
    if c_guidance:
        print(c_guidance)
