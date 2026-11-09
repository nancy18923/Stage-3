"""Polish Stage_3_Report.docx layout:
- put the Figure 2 caption on its own line (break instead of flowing beside the image)
- keep the 3.6 performance table and the 3.7 ablation table from splitting across pages
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import docx
from docx.shared import Pt

REPORT = os.path.join(os.path.dirname(PROJECT_ROOT), 'Stage_3_Report.docx')


def keep_table_together(table, keep_last=False):
    rows = table.rows
    last = len(rows) - 1
    for ri, row in enumerate(rows):
        if ri == last and not keep_last:
            continue
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True


def main():
    d = docx.Document(REPORT)

    # 1) rebuild the Figure 2 caption run with a leading line break
    for p in d.paragraphs:
        if any('Figure 2. Convergence curves' in r.text for r in p.runs):
            for r in list(p.runs):
                if r.text.startswith('Figure 2.'):
                    cap_text = r.text
                    r._r.getparent().remove(r._r)
                    cap = p.add_run()
                    cap.add_break()
                    cap.add_text(cap_text)
                    cap.font.size = Pt(9)
                    cap.font.italic = True
                    print('caption fixed')
                    break
            break

    # 2) keep the two result tables on one page
    perf = d.tables[2]
    abl = d.tables[3]
    keep_table_together(perf)
    keep_table_together(abl)
    # the intro paragraph right before each table should stick to its table
    for p in d.paragraphs:
        t = p.text.strip()
        if t.endswith('summarized below:') or t.endswith('not comparable with Section 3.6.'):
            p.paragraph_format.keep_with_next = True
    print('tables kept together')

    d.save(REPORT)
    print('saved', REPORT)


if __name__ == '__main__':
    main()
