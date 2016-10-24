%==============================
% EOSC350Lab.cls
% Provide EOSC350Lab.cls class
%==============================

% ==== Identification ==== %
\ProvidesClass{EOSC350Lab}
\NeedsTeXFormat{LaTeX2e}

\typeout{}
\typeout{ === EOSC 350  Lab Class (EOSC350Lab.cls) 2014 === }


% ==== Required Packages ====== %

\LoadClass[letterpaper, 12pt]{article}

\RequirePackage{fancyhdr}
\RequirePackage[top=1.2in,bottom=1.2in,left=1.1in,right=1.1in]{geometry}
\RequirePackage{graphicx}
\RequirePackage{empheq}
\RequirePackage{ifthen}
\RequirePackage{etoolbox}
\RequirePackage{color}
\RequirePackage{amsmath}
\RequirePackage{pdfpages}
\RequirePackage[titletoc]{appendix}
\RequirePackage{natbib}
\RequirePackage{amssymb}
\RequirePackage{makeidx}
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{epstopdf}
\RequirePackage{float}
\RequirePackage{mdwlist}
\RequirePackage{fullpage}
\RequirePackage{array}
\RequirePackage{booktabs}
\RequirePackage{framed}
\RequirePackage{nameref}
\definecolor{GIFblue}{rgb}{0,0,0.7}
%\RequirePackage[colorlinks=true,linkcolor=GIFblue,citecolor=GIFblue,urlcolor=GIFblue,ps2pdf,bookmarks=true]{hyperref}
\typeout{   Required packages loaded}

\pagestyle{empty}

% ===== OPTIONS ======= %
\newif\if@showAnswers

\DeclareOption{showAnswers}{\@showAnswerstrue}
\DeclareOption{twoside}{\PassOptionsToPackage{twoside}{article}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions

%=====================================================================
% Commands
%=====================================================================



\ifx\pdfoutput\undefined                         %LaTeX
  \RequirePackage[colorlinks=true,linkcolor=GIFblue,citecolor=GIFblue,urlcolor=GIFblue,ps2pdf,bookmarks=true]{hyperref}
\else                                            %PDFLaTeX
  \RequirePackage[colorlinks=true,linkcolor=GIFblue,citecolor=GIFblue,urlcolor=GIFblue,pdftex,bookmarks=true]{hyperref}
\pdfadjustspacing=1
\fi

% Lab info
\newcommand{\@TAname}{{}}
\newcommand{\@TAemail}{{}}
\newcommand{\@TAoffice}{{}}
\newcommand{\@labDate}{{}}
\newcommand{\@dueDate}{{}}
\newcommand{\@subtitle}{{}}
%\newcommand{\@showAnswers}{{}}


\newcommand{\TAname}[1]{\renewcommand{\@TAname}{#1}}
\newcommand{\TAemail}[1]{\renewcommand{\@TAemail}{#1}}
\newcommand{\TAoffice}[1]{\renewcommand{\@TAoffice}{#1}}
\newcommand{\labDate}[1]{\renewcommand{\@labDate}{#1}}
\newcommand{\dueDate}[1]{\renewcommand{\@dueDate}{#1}}
\newcommand{\DueDate}[1]{\renewcommand{\@dueDate}{#1}}
\newcommand{\LabDate}[1]{\renewcommand{\@labDate}{#1}}
\newcommand{\subtitle}[1]{\renewcommand{\@subtitle}{#1}}
\newcommand{\GPG}{\href{http://www.eos.ubc.ca/courses/eosc350/content/index.htm}{GPG} }

%\newboolean{showAnswersbool}
%\setboolean{showAnswersbool}{\@showAnswers}

\newcommand{\boxit}[1]{\fbox{#1}}
\newcommand{\figdir}[1]{\graphicspath{{#1/}}}
% Set up counters for questions and subsections

\newcounter{QuestionNum}
\newcounter{SubQuestionNum}[QuestionNum]

\renewcommand{\theQuestionNum}{\arabic{QuestionNum}}
\renewcommand{\theSubQuestionNum}{\alph{SubQuestionNum}}

\newcommand*{\question}[1]{\stepcounter{QuestionNum} %
   {\flushleft\hfill\begin{minipage}{\dimexpr\textwidth-0.5cm}
      \bigskip \textbf{Q\theQuestionNum.} #1 \\
      \end{minipage}}}

\renewcommand*{\part}[1]{\stepcounter{SubQuestionNum} %
  {\flushleft\hfill\begin{minipage}{\dimexpr\textwidth-1.25cm}
      \textbf{\theSubQuestionNum.} #1 \\
      \end{minipage}}}

%\ifthenelse{\boolean{\@showAnswers}}{%
\if@showAnswers%
  \newcommand*{\answer}[1]{%
    {\color{red} \textbf{A:} #1}
  }%
\else%
  \newcommand*{\answer}[1]%
    {{}}
\fi%


% Style for the enumerate enviornment
%\renewcommand{\theenumi}{(\alph{enumi})}
%\renewcommand{\theenumi}{\Roman{enumi}.)}
\renewcommand{\theenumi}{\arabic{enumi}.)}

\renewcommand{\labelenumi}{\theenumi}
\renewcommand{\theenumii}{\roman{enumii}}


% ==== DOCUMENT SETUP ====== %

\setlength{\headheight}{15pt}
\setlength{\headsep}{15pt}
\rhead{\small EOSC 350}\lhead{\small \@subtitle}
\lfoot{}\cfoot{\thepage}\rfoot{}
\pagestyle{fancy}

\AtBeginDocument{%
%\thispagestyle{empty}
\begin{flushleft}
{\Large \bf{\@title}}  \\
\bigskip
{\large TA:     \@TAname} \\
{\normalsize \quad \emph{e-mail:} \href{mailto:\@TAemail}{\@TAemail}}\\
{\normalsize \quad \emph{office: } \@TAoffice } \\
\bigskip
{\large DUE: \@dueDate} \\
\end{flushleft}
}%

\endinput
