# Tables

## Tables in Column

Use the following template for tables in column:

```latex
\begin{table}[htbp!]
\centering\small{ %\resizebox{\columnwidth}{!}{
\begin{tabular}{c||c|c|c} 
\bf A & \bf B & \bf C & \bf D \\
\hline\hline
0 & A0 & B0 & C0 \\
1 & A1 & B1 & C1 \\
2 & A2 & B2 & C2 \\
3 & A3 & B3 & C3 \\
\end{tabular}}
\caption{Description of this table.}
\label{tab:name}
\end{table}
```

* Use the following options for all tables:
  ```latex
  \begin{table}[htbp!]
  \centering\small{ %\resizebox{\columnwidth}{!}{
  ```
* If the table exceeds the column width, put the `tabular` inside a `\resizebox` instead:

  ```latex
  \centering\resizebox{\columnwidth}{!}{
  \begin{tabular}{c||c|c|c}
  ...
  \end{tabular}}
  ```
* Do not put any border around the table.
* Put double-lines to distinguish row (or/and) column headers, and use the bold font for the row header:
  ```
  \begin{tabular}{c||c|c|c} 
  \bf A & \bf B & \bf C & \bf D \\
  ```
* Make sure the label starts with the prefix `tab:`.


## Tables in Page

Use the following template for tables that expand to the full page.

```latex
\begin{table*}[htbp!]
\centering\small{ %\resizebox{\textwidth}{!}{
\begin{tabular}{c||c|c|c} 
\bf A & \bf B & \bf C & \bf D \\
\hline\hline
0 & A0 & B0 & C0 \\
1 & A1 & B1 & C1 \\
2 & A2 & B2 & C2 \\
3 & A3 & B3 & C3 \\
\end{tabular}}
\caption{Description of this table.}
\label{tab:name}
\end{table*}
```

* See the explanations for the [Tables in Column](#Tables-in-Column).
* If the table exceeds the page width, use `\textwidth` instead of `\columnwidth` for `\resizebox`.


## Sub-Tables

Use the following template to create sub-tables.

```latex
\begin{table}[htbp!]
\centering

\begin{subtable}{\columnwidth}
\centering\small{ %\resizebox{\columnwidth}{!}{
\begin{tabular}{c||c|c|c} 
\bf A & \bf B & \bf C & \bf D \\
\hline\hline
0 & A0 & B0 & C0 \\
1 & A1 & B1 & C1 \\
2 & A2 & B2 & C2 \\
3 & A3 & B3 & C3 \\
\end{tabular}}
\caption{Sub-table 1.}
\label{tab:subtable-1}
\end{subtable}
\vspace{0.5em}

\begin{subtable}{\columnwidth}
\centering\small{ %\resizebox{\columnwidth}{!}{
\begin{tabular}{c||c|c|c} 
\bf A & \bf B & \bf C & \bf D \\
\hline\hline
0 & A0 & B0 & C0 \\
1 & A1 & B1 & C1 \\
2 & A2 & B2 & C2 \\
3 & A3 & B3 & C3 \\
\end{tabular}}
\caption{Sub-table 2.}
\label{tab:subtable-2}
\end{subtable}

\caption{Description.}
\label{tab:name}
\end{table}
```

* The `subtable` environment requires the `subcaption` package.
* Put `\vspace{0.5em}` at the end of every sub-table expect for the very last one.
* See [Tables in Page](#Tables-in-Page) to create sub-tables that expand to the full page. 
