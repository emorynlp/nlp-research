# Paragraphs

Write each sentence in a separate line which makes it easier to comment out selective sentences.
For instance write as follows:

```latex
This is the first sentence.
This is the second sentence.
```

instead of writing as follows:

```latex
This is the first sentence. This is the second sentence.
```

which allows the following to comment out the second sentence:

```latex
This is the first sentence.
% This is the second sentence.
```

<!--
* Put `\noindent` on any paragraph that
  * Starts at the top of the page,
  * Follows tables, figures, or algorithm boxes.
* Avoid paragraphs ending with only few words; in other words, avoid any line including only few words.
-->
