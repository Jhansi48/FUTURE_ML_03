import os, glob, re, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.parser import parse_resume

def run_pii_audit():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    resumes_dir = os.path.join(base_dir, 'data', 'resumes')
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    resume_files = sorted(glob.glob(os.path.join(resumes_dir, '*.*')))
    assert len(resume_files) == 8, f'Expected 8 resumes, found {len(resume_files)}'
    
    formats = {os.path.splitext(f)[1].lower() for f in resume_files}
    pdf_pass = '.pdf' in formats
    docx_pass = '.docx' in formats
    txt_pass = '.txt' in formats
    
    known_name_tokens = [
        'alex', 'rivera', 'sarah', 'chen', 'david', 'kim', 'marcus', 'vance',
        'emily', 'watson', 'robert', 'kowalski', 'jessica', 'martinez', 'vikram', 'patel'
    ]
    
    names_masked = True
    emails_masked = True
    phones_masked = True
    urls_masked = True
    residual_pii_count = 0
    
    for f in resume_files:
        profile = parse_resume(f)
        anon = profile["raw_text"] # bounds check anonymized
        anon = profile["anonymized_text"].lower()
        
        # Check names
        for token in known_name_tokens:
            if re.search(r"\b" + re.escape(token) + r"\b", anon):
                names_masked = False
                residual_pii_count += 1
                print(f"[FAIL] Name token '{token}' found in {os.path.basename(f)}")
                
        # Check emails
        if "@" in anon and "[email_masked]" not in anon:
            emails_masked = False
            residual_pii_count += 1
            print(f"[FAIL] Email leak in {os.path.basename(f)}")
            
        # Check phones
        if re.search(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", anon):
            phones_masked = False
            residual_pii_count += 1
            print(f"[FAIL] Phone leak in {os.path.basename(f)}")
            
        # Check raw urls
        if re.search(r"https?://(?!.*_masked)", anon) or "linkedin.com/in/" in anon or "github.com/" in anon:
            urls_masked = False
            residual_pii_count += 1
            print(f"[FAIL] URL leak in {os.path.basename(f)}")

    overall = pdf_pass and docx_pass and txt_pass and names_masked and emails_masked and phones_masked and urls_masked and (residual_pii_count == 0)

    audit_text = f"""PII AUDIT ---------
Profiles checked: {len(resume_files)}
PDF: {'PASS' if pdf_pass else 'FAIL'}
DOCX: {'PASS' if docx_pass else 'FAIL'}
TXT: {'PASS' if txt_pass else 'FAIL'}
Names masked: {'PASS' if names_masked else 'FAIL'}
Emails masked: {'PASS' if emails_masked else 'FAIL'}
Phones masked: {'PASS' if phones_masked else 'FAIL'}
URLs masked: {'PASS' if urls_masked else 'FAIL'}
Residual PII detected: {residual_pii_count}
Overall: {'PASS' if overall else 'FAIL'}
"""
    print(audit_text)

    report_path = os.path.join(reports_dir, 'pii_audit_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(audit_text)
    return overall

if __name__ == '__main__':
    success = run_pii_audit()
    sys.exit(0 if success else 1)
