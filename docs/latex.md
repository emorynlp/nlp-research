Latex Conventions
=====

Use the following conventions when you write in LaTex.


## Packages

* Include the followings before `begin{document}`:
  ```latex
  \usepackage[hang,flushmargin]{footmisc}
  \usepackage{microtype}
  \usepackage{enumitem}
  \setenumerate[1]{leftmargin=*}
  \setitemize[1]{leftmargin=*}
  ```
  * `footmisc`: removes the indentation of footnotes.
  * `microtype`: adjusts margins between tokens (it makes the token alignment prettier).
  * `enumitem`: enables `setenumerate` and `setitemize` to remove the indentation of the `enumerate` and `itemize` environments.


## BibTex

* Any arXiv paper must be checked before citation whether or not it has been published to a peer-reviewed venue. If it has, use the reference from the peer-reviewed venue instead of arXiv.
* For the citation key, use the format `name:year` (e.g., `choi:19a`):
  * `name`: the last name of the first author in lowercase.
  * `year`: the last two digits of the published year followed by an alphabet to distinguish multiple publications with the same last name during the same year.
* For the title, surround the text with curly brackets; otherwise, the title will be lowercased in print:
  ```bibtex
  @inproceedings{choi:16a,
	Author = {Choi, Jinho D.},
	Title = {{Dynamic Feature Induction: The Last Gist to the State-of-the-Art}},
  ``` 
* For the series, use the format `acronym'year` (e.g., `ACL'19`), where `acronym` is the proper acronym of the venue and `year` is the last two digits of the published year.
* For the pages, make sure to put two dashes between the first and the last pages (e.g., `1--10`).


## Tables

* Use the options `[htbp!]` for all tables:
  ```latex
  \begin{table}[htbp!]
  ```
* 