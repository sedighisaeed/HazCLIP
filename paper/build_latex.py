"""Assemble 06_Paper/main.tex (IEEEtran journal mode) from MD sections.

Fixes over previous version:
- real IEEEtran class (tectonic fetches it) -> proper author block,
  two-column layout, roman-numbered sections, lettered subsections
- \\cite extracted BEFORE escaping -> citations render as [n] (was raw text!)
- markdown tables -> IEEE floats with \\caption (peek italic caption line),
  footnotesize, table* for wide ones
- $$..$$ blocks pulled before line-splitting; emitted as equation* + \\tag
- figures placed as [!t] floats inside their sections with real captions
"""

from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAP = os.path.join(ROOT, '06_Paper')

SECTIONS = [
    'Section1_Introduction_draft.md',
    'Section2_RelatedWork_draft.md',
    'Section3_Method_draft.md',
    'Section4_Experiments_draft.md',
    'Section5_Conclusion_draft.md',
]

UNI = [
    ('⁻⁵⁷', r'$^{-57}$'), ('⁻⁴⁶', r'$^{-46}$'), ('−', '-'),
    ('×', r'$\times$'),
    ('≤', r'$\leq$'), ('≥', r'$\geq$'), ('≈', r'$\approx$'),
    ('→', r'$\rightarrow$'), ('↔', r'$\leftrightarrow$'),
    ('✔', r'\cmark '), ('✘', r'\xmark '),
    ('β', r'$\beta$'), ('λ', r'$\lambda$'), ('τ', r'$\tau$'),
    ('α', r'$\alpha$'), ('σ', r'$\sigma$'),
    ('–', '--'), ('—', '---'), ('“', '``'), ('”', "''"),
    ('’', "'"), ('…', r'\ldots '), ('†', r'$\dagger$'),
    ('√', r'$\surd$'), ('·', r'$\cdot$'),
    ('±', r'$\pm$'), ('°', r'$^{\circ}$'),
]


def uni2tex(s: str) -> str:
    for a, b in UNI:
        s = s.replace(a, b)
    return s


def tex_escape(s: str) -> str:
    s = s.replace('\\', r'\textbackslash ')
    for ch in '&%$#_{}':
        s = s.replace(ch, '\\' + ch)
    return s


def md_inline(s: str) -> str:
    # 1) code spans
    codes = []
    def _code(m):
        codes.append(m.group(1))
        return f'\x00{len(codes)-1}\x00'
    s = re.sub(r'`([^`]+)`', _code, s)
    # 2) \cite BEFORE escaping (braces/backslash must survive verbatim)
    cites = []
    def _cite(m):
        cites.append(m.group(1))
        return f'\x03{len(cites)-1}\x03'
    s = re.sub(r'\\cite\{([^}]*)\}', _cite, s, flags=re.S)
    # 3) inline math $...$ BEFORE escaping
    inl = []
    def _inl(m):
        inl.append(m.group(1))
        return f'\x02{len(inl)-1}\x02'
    s = re.sub(r'\$([^$]+)\$', _inl, s)
    # 4) escape the plain text
    s = tex_escape(s)
    # 5) md emphasis
    s = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'\*([^*]+)\*', r'\\emph{\1}', s)
    # 6) restore verbatim pieces
    s = re.sub(r'R@(\d)', r'R@\1', s)
    for i, c in enumerate(codes):
        s = s.replace(f'\x00{i}\x00',
                      r'\texttt{' + c.replace('_', r'\_') + '}')
    for i, c in enumerate(cites):
        s = s.replace(f'\x03{i}\x03', r'\cite{' + c + '}')
    for i, m in enumerate(inl):
        s = s.replace(f'\x02{i}\x02', '$' + uni2tex(m) + '$')
    return uni2tex(s)


def md_to_tex(md: str) -> str:
    out = []
    open_list = [None]

    def close_list():
        if open_list[0]:
            out.append(r'\end{' + open_list[0] + '}')
            open_list[0] = None

    # pull multi-line $$..$$ blocks before splitting into lines
    block_out = []
    def _block(m):
        body = m.group(1).strip()
        block_out.append(r'\begin{equation*}' + body + r'\end{equation*}')
        return f'\x0E{len(block_out)-1}\x0E'
    md = re.sub(r'\$\$\s*(.*?)\s*\$\$', _block, md, flags=re.S)

    i, lines = 0, md.split('\n')
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                i += 1
            i += 1
            continue
        if ln.startswith('|') and i + 1 < len(lines) and \
                re.match(r'^\|[\s:|-]+\|$', lines[i + 1].strip()):
            close_list()
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in
                         lines[i].strip().strip('|').split('|')]
                rows.append(cells)
                i += 1
            header, body = rows[0], rows[2:]
            ncol = len(header)
            # caption: one or more consecutive lines "*Table X. ...*"
            caption = None
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith('*'):
                buf = []
                while j < len(lines) and lines[j].strip().startswith('*') \
                        and not lines[j].strip().endswith('*'):
                    buf.append(lines[j].strip())
                    j += 1
                if j < len(lines) and lines[j].strip().endswith('*'):
                    buf.append(lines[j].strip())
                    j += 1
                joined = ' '.join(buf)
                mm = re.match(r'^\*\s*((?:Table\s+[IVX]+\.\s*)?.*?)\s*\*\s*$',
                              joined, re.S)
                if mm and len(joined) > 15:
                    caption = mm.group(1)
                    i = j
            wide = ncol >= 6
            env = 'table*' if wide else 'table'
            out.append(r'\begin{' + env + '}[!t]')
            if caption:
                cap = re.sub(r'\s+', ' ', caption)
                out.append(r'\caption{' + md_inline(cap) + '}')
            out.append(r'\label{tab:' + re.sub(r'[^a-z0-9]+', '',
                       (caption or 't').lower()[:20]) + '}')
            out.append(r'\centering\footnotesize'
                       r'\setlength{\tabcolsep}{4pt}')
            out.append(r'\begin{tabular}{@{}' + 'c' * ncol + '@{}}'
                       + r'\toprule')
            out.append(' & '.join(r'\textbf{' + md_inline(c) + '}'
                                  for c in header) + r' \\')
            out.append(r'\midrule')
            for r in body:
                out.append(' & '.join(md_inline(c) for c in r) + r' \\')
            out.append(r'\bottomrule\end{tabular}')
            out.append(r'\end{' + env + '}')
            continue
        m = re.match(r'^(#{1,4})\s+(.*)$', ln)
        if m:
            close_list()
            lvl, txt = len(m.group(1)), m.group(2)
            # strip manual numbering: "1. ", "2.3. ", "A. "
            txt = re.sub(r'^\d+(\.\d+)*\.?\s+', '', txt)
            txt = re.sub(r'^[A-Z]\.\s+', '', txt)
            out.append((r'\section{' if lvl == 1 else r'\subsection{')
                       + md_inline(txt) + '}')
            i += 1
            continue
        if re.match(r'^\s*[-*]\s+\[ \]\s+', ln) or \
                re.match(r'^\s*[-*]\s+\[x\]\s+', ln, re.I):
            close_list()
            txt = re.sub(r'^\s*[-*]\s+\[.\]\s+', '', ln, flags=re.I)
            mark = r'$\boxtimes$' if '[x]' in ln.lower() else r'$\square$'
            out.append(r'\item[] ' + mark + '~' + md_inline(txt))
            i += 1
            continue
        if re.match(r'^\s*[-*]\s+', ln):
            if open_list[0] != 'itemize':
                close_list()
                out.append(r'\begin{itemize}')
                open_list[0] = 'itemize'
            out.append(r'\item ' + md_inline(re.sub(r'^\s*[-*]\s+', '', ln)))
            i += 1
            continue
        if re.match(r'^\s*\d+\.\s+', ln):
            if open_list[0] != 'enumerate':
                close_list()
                out.append(r'\begin{enumerate}')
                open_list[0] = 'enumerate'
            out.append(r'\item ' + md_inline(re.sub(r'^\s*\d+\.\s+', '', ln)))
            i += 1
            continue
        close_list()
        if ln.strip().startswith('>'):
            i += 1
            continue
        if ln.strip() == '---':
            i += 1
            continue
        line_tex = md_inline(ln)
        for bi in range(len(block_out)):
            line_tex = line_tex.replace(f'\x0E{bi}\x0E', block_out[bi])
        if line_tex.strip():
            out.append(line_tex)
        else:
            out.append('')
        i += 1
    close_list()
    res = '\n'.join(x for x in out if x != '')
    for bi in range(len(block_out)):
        res = res.replace(f'\x0E{bi}\x0E', block_out[bi])
    return res


TITLE = (r'\title{Haze-Robust Image Retrieval: Physics-Guided Adaptation '
         r'of Vision--Language Models for Content-Based Retrieval in '
         r'Hazy Domains}')

AUTHOR = r'''\author{\IEEEauthorblockN{Ali Hosseini}
\IEEEauthorblockN{Saeed Sedighi\textsuperscript{*}}
\IEEEauthorblockA{Department of Computer Engineering, Shi.C., Islamic Azad University, Shiraz, Iran\\
\{ali.hosseini7148, saeed.sedighi\}@iau.ac.ir}
\IEEEauthorblockA{\textsuperscript{*}Corresponding author: Saeed Sedighi (saeed.sedighi@iau.ac.ir)}
}'''

ABSTRACT = r'''\begin{abstract}
Content-based image retrieval fails under atmospheric haze: on a
severity-graded haze benchmark, R@1 of CLIP, SigLIP and DINOv2 collapses
by 50--65\% relative, while the same hazy queries still retrieve
same-severity hazy images perfectly --- revealing a \emph{query$\rightarrow$gallery
domain mismatch} rather than information loss. Guided by this diagnosis, we
present HazCLIP, a physics-guided adaptation of a frozen foundation
model that aligns clean and hazy views in a single embedding space.
HazCLIP couples (i) a hand-rolled LoRA on the last vision blocks, (ii) a
physics head supervised by atmospheric-scattering ground truth, and (iii) a
weather-conditioned prompt bank, trained with a three-term objective --- in
total 0.198\% trainable parameters, minutes on a laptop CPU.
HazCLIP recovers 73\% of the haze-induced retrieval loss on synthetic
benchmarks (extreme R@1: $0.353\rightarrow0.814\pm0.008$ across three
seeds), transfers to real photographs (Flickr8k: $0.297\rightarrow0.872$),
and generalizes to two \emph{independent real benchmarks without any
real-haze training}: ImageNet-C fog (zero-shot top-5 +15\% rel.) and a
1,623-pair real-haze corpus with instance-level relevance, where R@1 rises
from 0.811 to 0.930 ($-63\%$ relative miss rate; paired McNemar
$p=3.9\times10^{-57}$). Against the strongest two-stage baseline (DCP
dehazing $\rightarrow$ CLIP), HazCLIP matches extreme-severity R@1 while
\emph{preserving} clean-image performance, dominating all mAP metrics and
removing the dehazing pass; it even outperforms the most haze-robust raw
backbone (DINOv2) by +63\% relative --- evidence that targeted
physics-guided adaptation is a more efficient lever than backbone
replacement. Code, logs, and the evaluation protocol are released.
\end{abstract}

\begin{IEEEkeywords}
Image retrieval, haze, foundation models, LoRA, physics-guided learning,
domain alignment, CLIP.
\end{IEEEkeywords}'''

FIG1 = r'''\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig1_architecture.png}
\caption{HazCLIP architecture. A frozen CLIP ViT-B/32 is adapted with
Haze-LoRA (Q/V of the last four blocks, $r{=}8$). Hazy queries and clean
gallery images share the adapted encoder; an ASM supervision head
regresses severity $\hat{\beta}$ and transmission $\hat{t}$ from query
tokens, and a weather-conditioned prompt bank regularizes training. The
objective combines cross-domain InfoNCE with physics and invariance
terms; only 0.198\% of parameters are trainable and no restoration pass
is used at query time.}
\label{fig:arch}
\end{figure}'''

FIGS4 = r'''\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2_fm_severity.png}
\caption{Haze degrades all foundation models: R@1 of hazy queries against
a clean gallery (P1, class-level relevance, $n{=}600$) across ASM
severities.}
\label{fig:fm}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig3_p1_p2_paradox.png}
\caption{The domain-mismatch paradox: hazy$\rightarrow$clean retrieval
collapses with severity (P1) while hazy$\rightarrow$same-severity
retrieval is unaffected (P2) --- alignment, not information, is lost.}
\label{fig:paradox}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig4_head2head.png}
\caption{Head-to-head on one identical split (P1): raw CLIP vs.\ the
two-stage DCP$\rightarrow$CLIP baseline vs.\ HazCLIP.}
\label{fig:h2h}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{figures/fig5_realhaze_pairs.png}
\caption{Cumulative rank distribution on the real\_haze benchmark (all
1,623 real haze queries, gallery 1,623, exactly one relevant pair).}
\label{fig:realhaze}
\end{figure}'''

EPILOG = r'''
\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
'''


def main():
    body = [r'\documentclass[journal]{IEEEtran}',
            r'\usepackage{graphicx,booktabs,amsmath,amssymb,url}',
            r'\usepackage{pifont}',
            r'\newcommand{\cmark}{\ding{51}}',
            r'\newcommand{\xmark}{\ding{55}}',
            r'\usepackage[colorlinks=false,pdfborder={0 0 0}]{hyperref}',
            '',
            TITLE, '', AUTHOR, '',
            r'\begin{document}', '', r'\maketitle', '',
            ABSTRACT, '']
    for idx, f in enumerate(SECTIONS):
        p = os.path.join(PAP, f)
        with open(p) as fh:
            md = fh.read()
        if '\n---\n' in md:
            md = md.split('\n---\n')[0]
        body.append(md_to_tex(md))
        body.append('')
        if idx == 2:  # end of Method -> Fig. 1
            body.append(FIG1)
            body.append('')
        if idx == 3:  # end of Experiments -> Figs. 2-5
            body.append(FIGS4)
            body.append('')
    body.append(EPILOG)
    tex = '\n'.join(body)
    out = os.path.join(PAP, 'main.tex')
    with open(out, 'w') as fh:
        fh.write(tex)
    print('wrote', out, len(tex), 'chars')


if __name__ == '__main__':
    main()
