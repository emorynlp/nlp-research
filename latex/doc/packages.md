# Packages

Most 


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
