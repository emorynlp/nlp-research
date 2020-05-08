# References

## Bib Entries

Keep the following conventions to add entries in the bib file (e.g,. `acl2020.bib`):

### Preprint

Any preprint must be checked whether or not it has been published to a peer-reviewed venue. If it has, use the reference from the peer-reviewed venue instead of the preprint source such as arXiv.

### Key ###

Make sure there is no duplicate.

### Title ###

Surround the text with curly brackets; otherwise, the title will be lowercased in print:

```latex
@inproceedings{marcus-etal-1993-building,
Title = {{Building a Large Annotated Corpus of English: The Penn Treebank}},
```

```latex
@inproceedings{marcus-etal-1993-building,
Title = "{Building a Large Annotated Corpus of English: The Penn Treebank}",
```

### Booktitle / Journal ###

Do not use acronyms but the full venue names.
For instance, the following is good:

```
Booktitle = {Proceedings of the Annual Conference of the Association for Computational Linguistics},
```

whereas the following is bad:

```
Booktitle = {Proceedings of ACL},
```

### Series ###

Use the format `acronym'year` (e.g., `ACL'20`), where `acronym` is the acronym of the venue and `year` is the last two digits of the published year.

### Pages ###

Put two dashes between the first and the last pages (e.g., `1--10`).


## Citations

Use `\citet` when the reference is used in context:

```latex
\citet{marcus-etal-1993-building} introduced the Penn Treebank.
The Penn Treebank was introduced by \cite{marcus-etal-1993-building}.
```

Use `\cite` when the reference is used outside of the context:

```latex
The Penn Treebank was the first large corpus \cite{marcus-etal-1993-building}.
```

Use `\citealt` when the reference is used inside of parentheses:

```latex
The Penn Treebank (PTB; \citealt{marcus-etal-1993-building}) was the first large corpus.
```