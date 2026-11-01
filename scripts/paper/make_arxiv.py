#!/usr/bin/env python3
"""Build the arXiv submission package from paper/main.tex.

arXiv takes LaTeX SOURCE, not a PDF, and it rejects a package that reaches
outside itself. The CVPR source pulls figures from ../outputs/figures and
tables from ../outputs/tables, so the package has to be flattened. Doing that
as a script rather than by hand keeps one source of truth: the CVPR build and
the arXiv build are the same paper.

What it changes, and why each one matters to a moderator:

  \\pdfoutput=1 in the first five lines   forces pdfLaTeX; without it arXiv may
                                        pick a DVI path and fail on PDF figures
  [review] dropped from cvpr.sty        review mode prints "Anonymous CVPR
                                        submission" and line numbers. A preprint
                                        carries its authors
  ../ paths flattened                   arXiv refuses parent-directory escapes
  full-line comments stripped           the source is publicly downloadable;
                                        every draft note ships with it
  main.bbl copied in                    arXiv will not run BibTeX for you
  availability statement added          arXiv only; it names the repository,
                                        which must stay out of the blind CVPR
                                        submission

    python3 scripts/paper/make_arxiv.py
"""
import os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "paper")
OUT = os.path.join(ROOT, "paper", "arxiv")

AVAILABILITY = r"""
\paragraph{Code and data availability.} The measure, the evaluation harness,
the sealed split and its digest, and the amendment log that fixes every
hypothesis and acceptance threshold before its run are released at
\url{https://github.com/AKilalours/opendrivefm}, with an identical copy on
the second author's profile at
\url{https://github.com/AkilanManivannanak/opendrivefm}. Every number in this
paper is written by a named script to a JSON artifact in that repository, and
each table and figure names the artifact it reads.
"""


def strip_comments(text):
    """Drop whole-line comments only.

    Inline comments are left alone on purpose: a trailing % in LaTeX often
    suppresses a space, and removing it silently changes the typeset output.
    """
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("%"):
            continue
        out.append(line)
    return "\n".join(out)


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "figs"))
    os.makedirs(os.path.join(OUT, "tables"))

    tex = open(os.path.join(SRC, "main.tex")).read()

    # 1. pdfLaTeX, declared in the first five lines
    tex = "\\pdfoutput=1\n" + tex

    # 2. a preprint is not anonymous
    tex = tex.replace("\\usepackage[review]{cvpr}", "\\usepackage{cvpr}")

    # 3. flatten every path that leaves the package
    tex = tex.replace("\\graphicspath{{../outputs/figures/}}",
                      "\\graphicspath{{figs/}}")
    tex = re.sub(r"\\input\{\.\./outputs/tables/([^}]+)\}",
                 r"\\input{tables/\1}", tex)
    tex = tex.replace("\\bibliography{../opendrivefm_refs}",
                      "\\bibliography{opendrivefm_refs}")

    # 4. the availability statement, arXiv only
    tex = tex.replace("\\section{Conclusion}", AVAILABILITY + "\n\\section{Conclusion}")

    tex = strip_comments(tex)
    open(os.path.join(OUT, "main.tex"), "w").write(tex)

    # 5. everything the package needs, copied in
    for f in ("cvpr.sty", "ieeenat_fullname.bst", "main.bbl"):
        shutil.copy(os.path.join(SRC, f), OUT)
    shutil.copy(os.path.join(ROOT, "opendrivefm_refs.bib"), OUT)
    figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    for f in sorted(set(figs)):
        src = os.path.join(ROOT, "outputs/figures", f)
        if not os.path.exists(src):
            print(f"  MISSING FIGURE {f}"); continue
        shutil.copy(src, os.path.join(OUT, "figs", f))
    for t in re.findall(r"\\input\{tables/([^}]+)\}", tex):
        name = t if t.endswith(".tex") else t + ".tex"
        shutil.copy(os.path.join(ROOT, "outputs/tables", name),
                    os.path.join(OUT, "tables", name))

    # 6. compile it exactly as arXiv will
    env = dict(os.environ, TEXINPUTS=f"{OUT}:")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=OUT, capture_output=True)
    log = open(os.path.join(OUT, "main.log"), errors="ignore").read()
    bad = [l for l in log.split("\n")
           if "Undefined control sequence" in l or "undefined" in l.lower()
           or "LaTeX Error" in l]
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(OUT) for f in fs
               if not f.endswith((".log", ".aux", ".out", ".brf", ".blg")))
    pages = ""
    try:
        info = subprocess.run(["pdfinfo", os.path.join(OUT, "main.pdf")],
                              capture_output=True, text=True).stdout
        pages = [l for l in info.split("\n") if l.startswith("Pages")][0]
    except Exception:
        pass
    print(f"package: {os.path.relpath(OUT, ROOT)}")
    print(f"  {pages}")
    print(f"  upload size {size/1e6:.1f} MB")
    print(f"  figures {len(set(figs))}, tables inlined")
    print("  WARNINGS:" if bad else "  no undefined references or LaTeX errors")
    for b in bad[:8]:
        print("   ", b.strip()[:110])

    # 7. the upload itself: source only. arXiv builds the PDF, and shipping
    #    main.pdf alongside the source is what gets a submission flagged.
    tar = os.path.join(ROOT, "paper", "arxiv-submission.tar.gz")
    keep = ["main.tex", "cvpr.sty", "ieeenat_fullname.bst", "main.bbl",
            "opendrivefm_refs.bib", "figs", "tables"]
    if os.path.exists(tar):
        os.remove(tar)
    subprocess.run(["tar", "-czf", tar, "-C", OUT] + keep, check=True)
    print(f"  upload tarball {os.path.relpath(tar, ROOT)} "
          f"({os.path.getsize(tar)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
