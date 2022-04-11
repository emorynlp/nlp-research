# Tables

## Tables in Column

Use the following template to include tables in column:

```latex
\begin{table}[htbp!]
\centering\small{ %\resizebox{\columnwidth}{!}{
\begin{tabular}{c||c|c|c} 
\toprule
\bf A & \bf B & \bf C & \bf D \\
\midrule
0 & A0 & B0 & C0 \\
1 & A1 & B1 & C1 \\
2 & A2 & B2 & C2 \\
3 & A3 & B3 & C3 \\
\bottomrule
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
* Make sure values in the header row are always center aligned regardless of the configuration.  You can specify this by using `\multicolumn`:
  ```latex
  \begin{tabular}{c||r|r|r} 
  \bf A & \multicolumn{1}{c|}{\bf B} & \multicolumn{1}{c|}{\bf C} & \multicolumn{1}{c}{\bf D} \\
  ```
* Make sure the label starts with the prefix `tab:`.


## Tables in Page

Use the following template to include tables that expand to the full page.

```latex
\begin{table*}[htbp!]
\centering\small{ %\resizebox{\textwidth}{!}{
\begin{tabular}{c||c|c|c} 
\toprule
\bf A & \bf B & \bf C & \bf D \\
\midrule
0 & A0 & B0 & C0 \\
1 & A1 & B1 & C1 \\
2 & A2 & B2 & C2 \\
3 & A3 & B3 & C3 \\
\bottomrule
\end{tabular}}
\caption{Description of this table.}
\label{tab:name}
\end{table*}
```

* If the table exceeds the page width, use `\textwidth` instead of `\columnwidth` for `\resizebox`.
* See the other explanations for the [Tables in Column](#Tables-in-Column).


## Sub-Tables

Use the following template to create sub-tables.

```latex
\begin{table}[htbp!]
\centering

\begin{subtable}{\columnwidth}
\centering\small{ %\resizebox{\columnwidth}{!}{
\begin{tabular}{c||c|c|c} 
\toprule
\bf A & \bf B & \bf C & \bf D \\
\midrule
0 & A0 & B0 & C0 \\
1 & A1 & B1 & C1 \\
2 & A2 & B2 & C2 \\
3 & A3 & B3 & C3 \\
\bottomrule
\end{tabular}}
\caption{Sub-table 1.}
\label{tab:name-1}
\end{subtable}
\vspace{0.5em}

\begin{subtable}{\columnwidth}
\centering\small{ %\resizebox{\columnwidth}{!}{
\begin{tabular}{c||c|c|c} 
\toprule
\bf A & \bf B & \bf C & \bf D \\
\midrule
0 & A0 & B0 & C0 \\
1 & A1 & B1 & C1 \\
2 & A2 & B2 & C2 \\
3 & A3 & B3 & C3 \\
\bottomrule
\end{tabular}}
\caption{Sub-table 2.}
\label{tab:name-2}
\end{subtable}

\caption{Description.}
\label{tab:name}
\end{table}
```

* The `subtable` environment requires the `subcaption` package.
* Put `\vspace{0.5em}` at the end of every sub-table expect for the very last one.
* See [Tables in Page](#Tables-in-Page) to create sub-tables that expand to the full page. 
