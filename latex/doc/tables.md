# Tables

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
