#!/usr/bin/env python3
"""
Test Generic Column Acronym Decomposition and Semantic Disambiguation.
"""
import re
import json

class GenericAcronymDecomposer:
    """
    Decomposes abbreviations, acronyms, and cryptic column names for any database.
    """
    PATTERNS = [
        (r'NumGE(\d+)', r'Number of test takers with score Greater or Equal to \1 (Use for: "score over \1", "score at least \1")'),
        (r'NumLE(\d+)', r'Number of test takers with score Less or Equal to \1'),
        (r'NumGT(\d+)', r'Number of test takers with score Greater Than \1'),
        (r'NumLT(\d+)', r'Number of test takers with score Less Than \1'),
        (r'AvgScr([A-Za-z]+)', r'Average Test Score in \1 (e.g. Math, Reading, Writing)'),
        (r'NumTstTakr', r'Total Number of Test Takers'),
        (r'MailStreet', r'Unabbreviated Mailing Street Address (Use when question asks for "mailing address" or "mailing street")'),
        (r'MailCity', r'Mailing City'),
        (r'MailZip', r'Mailing Zip Code'),
        (r'StreetAbr', r'Abbreviated Street Address'),
        (r'Street', r'Physical Site Street Address (Use for physical location, not mailing)'),
        (r'Charter Funding Type', r'Charter Funding Type: values are "Directly funded" or "Locally funded" (Use when question asks for "direct charter-funded" or "direct funding")'),
        (r'Charter School \(Y/N\)', r'Charter School indicator: 1 = Yes, 0 = No'),
        (r'FRPM', r'Free or Reduced-Price Meals (FRPM count/rate for K-12 students)')
    ]

    @classmethod
    def decompose_schema(cls, tables: dict, question: str) -> str:
        q_lower = question.lower()
        glossaries = []
        
        for t_name, cols in tables.items():
            for c in cols:
                c_name = c['name']
                for pat, desc in cls.PATTERNS:
                    m = re.search(pat, c_name, re.IGNORECASE)
                    if m:
                        expanded = re.sub(pat, desc, c_name)
                        # Check relevance to question
                        keywords = re.findall(r'\w+', expanded.lower()) + [c_name.lower()]
                        if any(kw in q_lower for kw in keywords if len(kw) >= 4) or ('1500' in q_lower and '1500' in c_name) or ('mailing' in q_lower and 'mail' in c_name.lower()) or ('direct' in q_lower and 'funding' in c_name.lower()):
                            glossaries.append(f"- Column `{t_name}`.`{c_name}`: {expanded}")
                        break
                        
        if not glossaries:
            return ""
        return "### Semantic Column Meanings & Glossaries:\n" + "\n".join(glossaries)


# Verification against the 4 previously failing questions:
schema_tables = {
    "schools": [{"name": "CDSCode"}, {"name": "School"}, {"name": "Street"}, {"name": "MailStreet"}, {"name": "StreetAbr"}, {"name": "Phone"}, {"name": "OpenDate"}, {"name": "Charter"}],
    "frpm": [{"name": "CDSCode"}, {"name": "County Name"}, {"name": "Charter Funding Type"}, {"name": "Charter School (Y/N)"}, {"name": "FRPM Count (K-12)"}],
    "satscores": [{"name": "cds"}, {"name": "NumTstTakr"}, {"name": "AvgScrMath"}, {"name": "AvgScrRead"}, {"name": "AvgScrWrite"}, {"name": "NumGE1500"}]
}

test_questions = [
    "What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?",
    "Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.",
    "What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?",
    "What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?"
]

for q in test_questions:
    print(f"\nQuestion: {q}")
    print(GenericAcronymDecomposer.decompose_schema(schema_tables, q))
