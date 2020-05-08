# Figures

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
