# Figures

## Image Format

Before including figures, make sure the followings:

* Export all images to the PDF format. 
  Converting image files (e.g., PNG) to PDF does not render well; you must export the vectorized images to PDF.
* Crop all white margins around the image. 
  If you use Mac OS or Linux, you can simple type the following command in a terminal, which will crop `image.pdf` and save it to `image-crop.pdf`:
  ```bash
  $ pdfcrop image.pdf image-crop.pdf
  ```

## Figures in Column

Use the following template to include figures in column:

```latex
\begin{figure}[htbp!]
\centering
\includegraphics[width=\columnwidth]{img/image.pdf}
\caption{Description.}
\label{img:name}
\end{figure}
```

* Use the following options for all figures:
  ```latex
  \begin{figure}[htbp!]
  \centering
  ```
* If the figure seems too large, use `scale` instead of `width` as the option for `\includegraphics`:
  ```latex
  \includegraphics[scale=0.5]{img/image.pdf}
  ```
* Make sure the label starts with the prefix `fig:`.


## Figures in Page

Use the following template to include tables that expand to the full page.

```latex
\begin{figure*}[htbp!]
\centering
\includegraphics[width=\textwidth]{img/image.pdf}
\caption{Description.}
\label{img:name}
\end{figure*}

```

* Use the `\textwidth` option instead of `\columnwidth` for `\resizebox`.
* See the other explanations for the [Figures in Column](#Figures-in-Column).


## Sub-Figures

Use the following template to create sub-figures.

```latex
\begin{figure}[htbp!]
\centering

\begin{subfigure}{\columnwidth}
\centering
\includegraphics[width=\columnwidth]{img/image.pdf}
\caption{Sub-figure 1.}
\label{tab:name-1}
\end{subfigure}

\begin{subfigure}{\columnwidth}
\centering
\includegraphics[width=\columnwidth]{img/image.pdf}
\caption{Sub-figure 1.}
\label{tab:name-1}
\end{subfigure}

\caption{Description.}
\label{tab:name}
\end{figure}
```

* The `subfigure` environment requires the `subcaption` package.
* Put `\vspace{0.5em}` at the end of every sub-figure expect for the very last one.
* See [Figures in Page](#Figures-in-Page) to create sub-tables that expand to the full page. 
