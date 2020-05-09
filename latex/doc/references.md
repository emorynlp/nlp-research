# References

## Bibliography Style

Most LaTex templates come with their own style files.
Our [template](https://www.overleaf.com/read/xqgmwyqbqxbz) uses the style file, `acl_natbib.bst`, that is indicated towards the bottom of main tex file, `acl2020.tex`:

```latex
\bibliographystyle{acl_natbib}
```

## Bib Entries

Create a bib file to add references used in your paper.
Our [template](https://www.overleaf.com/read/xqgmwyqbqxbz) uses bib file, `acl2020.bib`, that is indicated towards the bottom of main tex file, `acl2020.tex`:

```latex
\bibliography{acl2020}
```

Any preprint must be checked whether or not it has been published to a peer-reviewed venue. If it has, use the reference from the peer-reviewed venue instead of the preprint source such as arXiv.

Keep the following conventions to add entries in the bib file:

* For `key`, make sure there is no duplicate.
* For `title`, surround the text with curly braces; otherwise, the title will be lowercased in print:
  ```
  @inproceedings{marcus-etal-1993-building,
      Title = {{Building a Large Annotated Corpus of English: The Penn Treebank}},
  ```
  ```
  @inproceedings{marcus-etal-1993-building,
      Title = "{Building a Large Annotated Corpus of English: The Penn Treebank}",
  ```
* For `booktitle` or `journal`, do not use acronyms but the full venue names. For instance, the following is good:
  ```
  Booktitle = {Proceedings of the Annual Conference of the Association for Computational Linguistics},
  ```
  whereas the following is bad:
  ```
  Booktitle = {Proceedings of ACL},
  ```
* For `series`, use the format `acronym'year` (e.g., `ACL'20`), where `acronym` is the acronym of the venue and `year` is the last two digits of the published year.
* For `pages`, put two dashes between the first and the last pages (e.g., `1--10`).
* For `url`, add the link to the original source of the paper (e.g., [ACL Anthology](https://www.aclweb.org/anthology/)).


## Citations

Use `\citet` when the reference is used in context:

```latex
\citet{devlin-etal-2019-bert} introduced BERT.
BERT was introduced by \cite{devlin-etal-2019-bert}.
```

Use `\cite` when the reference is used outside of the context:

```latex
BERT is a transformer-based contextualized embedding model \cite{devlin-etal-2019-bert}.
```

Use `\citealt` when the reference is used inside of parentheses:

```latex
Bidirectional Encoder Representations from Transformers (BERT; \citealt{devlin-etal-2019-bert}) is used to generate token embeddings.
```