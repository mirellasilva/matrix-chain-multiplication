%%
%% This is file `utfpr-pg.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% utfpr-pg.dtx  (with options: `class')
%% ----------------------------------------------------------------
%% utfpr-pg --- Classe para trabalhos acadêmicos da UTFPR-PG
%% Author:  Fabiano Rosas, Gabriel B. Casella
%% E-mail:  fabianorosas@gmail.com, gbc921@gmail.com
%% License: Released under the LaTeX Project Public License v1.3c or later
%% See:     http://www.latex-project.org/lppl.txt
%% ----------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{utfpr-pg}
%%[2013/10/30 v0.9.0 Classe para trabalhos acadêmicos da UTFPR-PG]
[2014/11/13 v1.0.0 Classe para trabalhos acadêmicos da UTFPR-PG]
%%    entradas do sumário com formatação correta (exceto capitalização)
%%    referência antes do resumo
%%    folha de rosto de acordo com a norma
%%    opções da classe não são mais necessárias no .tex do usuário
%%    remoção do header das páginas
%%    ajuste de espaçamentos
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{abntex2}}
\ProcessOptions\relax
\LoadClass[a4paper,12pt,openright,oneside,chapter=TITLE,section=TITLE,brazil]{abntex2}
\RequirePackage[alf,abnt-emphasize=bf,abnt-repeated-author-omit=yes,abnt-etal-text=it]{abntex2cite}
\RequirePackage{indentfirst}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lastpage}
\RequirePackage{etoolbox}
\RequirePackage{multibib}
\RequirePackage[within=none]{newfloat}
\newcommand*\erro[1]{\@latex@error{Defina \noexpand#1!}\@ehc}
\newcommand\utfpr{Universidade Estadual de Campinas}
\newcommand\futpr{Federal University of Technology - Paraná}

\renewcommand\imprimirtitulo{\erro\titulo}
\renewcommand*\titulo[1]{\renewcommand{\imprimirtitulo}{#1}}

\providecommand\imprimircurso{\erro\curso}
\newcommand*\curso[1]{\renewcommand{\imprimircurso}{#1}}

\providecommand\imprimirdepartamento{}
\newcommand*\departamento[1]{\renewcommand{\imprimirdepartamento}{#1}}

\providecommand\theschool{\erro\school}
\newcommand*\school[1]{\renewcommand{\theschool}{#1}}
\DeclareFloatingEnvironment[
    fileext=loq,
    listname=Lista de Quadros,
    name=Quadro,
    placement=tbhp,
]{quadro}
 \setlrmarginsandblock{3cm}{2cm}{*}
 \setulmarginsandblock{3cm}{2cm}{*}
 \checkandfixthelayout
\renewcommand{\imprimircapa}{
        \begin{capa}
          \center
          {\large\MakeUppercase\utfpr}\par
          {\large\MakeUppercase\imprimirdepartamento}\par
          {\large\MakeUppercase\imprimircurso}\par
          \vfill
          {\large\MakeUppercase\imprimirautor}\par
          \vfill
          {\bfseries\large\imprimirtitulo}\par
          \vfill
          {\large\MakeUppercase\imprimirtipotrabalho}\par
          \vfill
          {\large\MakeUppercase\imprimirlocal}\par
          {\large\imprimirdata}\par
        \end{capa}
}
\renewcommand{\folhaderostocontent}{
  \begin{center}

    {\large\MakeUppercase\imprimirautor}

    \vspace*{\fill}
    \vspace*{\fill}
    \begin{center}
      {\bfseries\large\MakeUppercase\imprimirtitulo}\par
    \end{center}
    \vspace*{\fill}

    \abntex@ifnotempty{\imprimirpreambulo}{%
      \hspace{.45\textwidth}
      \begin{minipage}{.5\textwidth}
       \SingleSpacing
         \imprimirpreambulo
         \vskip\onelineskip
         {\imprimirorientadorRotulo~\imprimirorientador\par}
         \abntex@ifnotempty{\imprimircoorientador}{%
                 {\imprimircoorientadorRotulo~\imprimircoorientador}}%
       \end{minipage}%
    }%

    \vspace*{\fill}
    \vspace*{\fill}

    {\large\MakeUppercase\imprimirlocal}\par
    \par
    {\large\imprimirdata}
    \vspace*{1cm}

  \end{center}
}
\RequirePackage{tgtermes}
\RequirePackage[font=small,font=bf,labelfont=bf]{caption}
\renewcommand{\ABNTEXchapterfont}{\bfseries}
\renewcommand{\ABNTEXchapterfontsize}{\normalsize}
\renewcommand{\ABNTEXsectionfont}{}
\renewcommand{\ABNTEXsectionfontsize}{\normalsize}
\renewcommand{\ABNTEXsubsectionfontsize}{\normalsize}
\renewcommand{\ABNTEXsubsubsubsectionfont}{\itshape}
\apptocmd{\pretextual}{%
        \SingleSpacing}{}{%
        \ClassWarning{utfpr-pg}{%
                Erro ao modificar o comando \@backslashchar pretextual,
                isto pode ter causado algum erro de formatação no seu documento.
                Consulte os desenvolvedores da classe.}}
\apptocmd{\textual}{%
        \pagestyle{simple}%
        \aliaspagestyle{chapter}{simple}%
        \OnehalfSpacing}{}{%
        \ClassWarning{utfpr-pg}{%
                Erro ao modificar o comando \@backslashchar textual,
                isto pode ter causado algum erro de formatação no seu documento.
                Consulte os desenvolvedores da classe.}}
\setlength\beforechapskip{2.0\onelineskip}
\setlength\afterchapskip{2.0\onelineskip}

\setlength\beforesecskip{\beforechapskip}
\setlength\aftersecskip{\afterchapskip}

\setlength\beforesubsecskip{\beforechapskip}
\setlength\aftersubsecskip{\afterchapskip}

\setlength\beforesubsubsecskip{\beforechapskip}
\setlength\aftersubsubsecskip{\afterchapskip}
\setlength\parindent{1.5cm}
\renewcommand\numberlinehook[1]{%
        \addtolength{\@tempdima}{\widthof{#1}}
}
\renewcommand\chapternumberlinehook\numberlinehook
\renewcommand{\tocprintchapter}{
       \addtocontents{toc}{\cftsetindents{chapter}{0em}{1em}}}
\cftsetindents{section}{0em}{1em}
\cftsetindents{subsection}{0em}{1em}
\cftsetindents{subsubsection}{0em}{1em}
\cftsetindents{paragraph}{0em}{1em} %% use paragraph instead of subsubsubsection
\setlength\cftbeforechapterskip{0cm}
\renewcommand{\cftchapterleader}{\normalfont\cftdotfill{\cftchapterdotsep}}
\renewcommand{\cftdotsep}{1}

\newcommand{\upcase}[1]{\uppercase{#1}}

\renewcommand{\cftchapterfont}{\normalfont\bfseries}
\renewcommand{\cftsectionfont}{\mdseries\upcase}
\renewcommand{\cftsubsectionfont}{\normalfont}
\renewcommand{\cftsubsubsectionfont}{\normalfont}
\renewcommand{\cftparagraphfont}{\normalfont\itshape}

\renewcommand{\cftchapterpagefont}{\cftchapterfont}
\renewcommand{\cftsectionpagefont}{\cftsectionfont}
\renewcommand{\cftsubsectionpagefont}{\cftsubsectionfont}
\renewcommand{\cftsubsubsectionpagefont}{\cftsubsubsectionfont}
\renewcommand{\cftparagraphpagefont}{\cftparagraphfont}
\nobibintoc
\renewcommand{\prebibhook}{%
        \addtocontents{toc}{\cftsetindents{chapter}{.27em}{0em}}
        \addcontentsline{toc}{chapter}{\upcase\bibname}}
\newcites{autocitacaopt}{ }
\newcites{autocitacaoen}{ }
\AfterEndPreamble{
  \edef\autocitefile{autocitacao.bib}
  \IfFileExists{\autocitefile}{
    \typeout{File autocitacao.bib exists. Using it.}
  }{
    \typeout{File autocitacao.bib does not exist. Creating a new one.}
    \newwrite\@autocite
    \immediate\openout\@autocite=autocitacao.bib\relax
    \write\@autocite{%
      @thesis{autocitacaoen,
        author={\imprimirautor},
        title = {\thetitle},
        type={\imprimirtipotrabalho\ (\imprimircurso)},
        school={\theschool},
        pages={\pageref{LastPage}},
        pagename={f.},
        year={\imprimirdata},
        type={\imprimirtipotrabalho}
      }
    }
    \write\@autocite{%
      @thesis{autocitacaopt,
        author={\imprimirautor},
        title = {\imprimirtitulo},
        type={\imprimirtipotrabalho\ (\imprimircurso)},
        school={\imprimirinstituicao},
        pages={\pageref{LastPage}},
        pagename={f.},
        year={\imprimirdata},
        type={\imprimirtipotrabalho}
      }
    }
  }
}
\newcommand\autocitacao[1][pt]{
  \renewcommand{\prebibhook}{}
  \begingroup
  \let\clearpage\relax
  \vspace{-40pt}
  \expandafter\csname nociteautocitacao#1\endcsname{autocitacao#1}
  \let\oldcite\nocite
  \renewcommand\nocite\citeoption
  \expandafter\csname nociteautocitacao#1\endcsname{abnt-emphasize=bf,abnt-full-initials=yes,abnt-thesis-year=both}
  \renewcommand\nocite\oldcite
  \expandafter\csname bibliographystyleautocitacao#1\endcsname{abntex2-\AbntCitetype}
  \expandafter\csname bibliographyautocitacao#1\endcsname{\autocitefile}
  \vspace{\onelineskip}
  \endgroup
}
%% Customizations of the abnTeX2 class (http://abnTeX2.googlecode.com)
%% 
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License (LPPL), either
%% version 1.3c of this license or (at your option) any later
%% version.  The latest version of this license is in the file:
%% 
%% http://www.latex-project.org/lppl.txt
%% 
%% This work is "maintained" (as per LPPL maintenance status) by
%% (not set).
%% 
%% This work consists of the file utfpr-pg.dtx and a Makefile.
%% Running make generates the derived files README.txt, utfpr-pg.pdf and utfpr-pg.cls.
%% Running make inst installs the files in the user's TeX tree.
%% Running make install installs the files in the local TeX tree.
%% 
%% Further information about abnTeX2 is available on http://abntex2.googlecode.com/
%%
%% End of file `utfpr-pg.cls'.
