# Packages

## Required Packages

Most LaTex templates come with their own style files.
Our [template](https://www.overleaf.com/read/xqgmwyqbqxbz) uses the style file, `acl2020.sty`, that is indicated as the first package:

```latex
\usepackage[hyperref]{acl2020}
```

The following packages (and definition) are required by ACL:

```latex
\usepackage[hyperref]{acl2020}
\usepackage{times}
\usepackage{latexsym}
\renewcommand{\UrlFont}{\ttfamily\small}
```

## Recommended Packages

Include the following packages (and definition) to the main tex file (e.g,. `acl2020.tex`):

```latex
\usepackage{microtype}      % make text margins prettier
\usepackage{graphicx}       % import graphical images
\usepackage[T1]{fontenc}    % fix font related glitches such as "|"
\usepackage{multirow}       % merge rows in tables
\usepackage{subcaption}     % create sub-tables and sub-figures
\usepackage{amssymb}        % enable \mathbb, \mathcal
\usepackage{bold-extra}     % enable \texttt{\textbf{}}
\usepackage{bm}             % enable bold font in math mode
\usepackage[hang,flushmargin]{footmisc}  % minimize footnote indentation
\newcommand{\LN}{\linebreak\noindent}    % to manage inline spacing
```

## Optional Packages

The following packages provide useful fonts:

```
\usepackage{amsfonts}
\usepackage{amsmath}
```

Include the followings if you want no indentation for the enumerate/itemize environments:

```latex
\usepackage{enumitem}
\setenumerate[1]{leftmargin=*}    % no enumerate indentation
\setitemize[1]{leftmargin=*}      % no itemize indentation
```
