LaTex Conventions
=====

Use the following conventions when you write in LaTex.


## Packages

* Include the followings before `\begin{document}`:
  ```latex
  \usepackage[hang,flushmargin]{footmisc}
  \usepackage{microtype}
  \usepackage{graphicx}
  \usepackage{enumitem}
  \setenumerate[1]{leftmargin=*}
  \setitemize[1]{leftmargin=*}
  ```
  * `footmisc`: removes the indentation of footnotes.
  * `graphics`: enables to include PDF images.
  * `microtype`: adjusts margins between tokens (it makes the token alignment prettier).
  * `enumitem`: enables `setenumerate` and `setitemize` to remove the indentation of the `enumerate` and `itemize` environments.


## Bib Entries

* Any arXiv paper must be checked whether or not it has been published to a peer-reviewed venue. If it has, use the reference from the peer-reviewed venue instead of arXiv.
* For the citation key, use the format `name:year` (e.g., `choi:19a`):
  * `name`: the last name of the first author in lowercase.
  * `year`: the last two digits of the published year followed by an alphabet to distinguish multiple publications with the same last name during the same year.
* For the title, surround the text with curly brackets; otherwise, the title will be lowercased in print:
  ```latex
  @inproceedings{choi:16a,
  Author = {Choi, Jinho D.},
  Title = {{Dynamic Feature Induction: The Last Gist to the State-of-the-Art}},
  ...
  ```
* For the series, use the format `acronym'year` (e.g., `ACL'19`), where `acronym` is the acronym of the venue and `year` is the last two digits of the published year.
* For the pages, make sure to put two dashes between the first and the last pages (e.g., `1--10`).

## References

- Use `\citet` when the reference is used in context:

- ```
  \citet{marcus:93a} introduced the Penn Treebank.
  The Penn Treebank was introduced by \cite{marcus:93a}.
  ```

- Use `\cite` when the reference is used outside of the context:

  ```
  The Penn Treebank was the first large corpus~\cite{marcus:93a}.
  ```

## Tables

* Use the options `[htbp!]`, and put `\centering` and `\small` for all tables:
  ```latex
  \begin{table}[htbp!]
  \centering\small
  ```

* Do not put borders around the table. Put double-lines to distinguish row (or/and) column headers, and use the bold font for the row header:

  ![image-20190311150224433](img/latex-table.png)

* If the table exceeds the column width, put the `tabular` inside a `\resizebox`:

  ```latex
  {\resizebox{\columnwidth}{!}{
  \begin{tabular}{c||c|c|c}
  ...
  \end{tabular}}
  ```

  * Use `\columnwidth` to fit in a single column.
  * Use `\textwidth` to fit in a page. 

## Figures

- Use the options `[htbp!]` and put `\centering` for all tables:

  ```latex
  \begin{figure}[htbp!]
  \centering
  ```

- Include images in the PDF format. Crop all white spaces around the image contents.

- If the image exceeds the column width, adjust the image width with `\columnwidth` or `\textwidth`:

- ```latex
  \includegraphics[width=\columnwidth]{img/diagram.pdf}
  ```

## Labels

* Labels must start with
  * `sec:`, `ssec:`, `sssec:` for sections, subsections, and subsubsections (e.g., `sec:introduction`),
  * `tbl:` for tables (e.g., `tbl:stats`),
  * `fig:` for figures (e.g., `fig:diagram`).

## Paragraphs

* Put `\noindent` on any paragraph that
  * Starts at the top of the page,
  * Follows tables, figures, or algorithm boxes.
* Avoid paragraphs ending with only few words; in other words, avoid any line including only few words.

