%{
    #include "astnode.hpp"
    #include "parse.hpp"

    #define YY_USER_ACTION yylloc.first_line = yylloc.last_line = yylineno;

    #ifdef __GNUC__
    static void yyunput(int c, register char * yy_bp) __attribute__((unused));
    #endif
%}

%option yylineno

eventually          ("ev"|"eventually")
always              ("alw"|"always")
lesseq              "<="
greatereq           ">="
comment             #[^\n]*
letter              [a-zA-Z]
digit               [0-9]
float               {digit}+.{digit}*
identifier_body     ({letter}|{digit})
identifier          {letter}{identifier_body}*
whitespace          [ \t\n]
lexem               [_()<>,\]\[]

    int errors = 0;
%%

{eventually}            return EV;
{always}                return ALW;
until                   return UNTIL;
not                     return NOT;
and                     return AND;
or                      return OR;
{lesseq}                return LESSEQ;
{greatereq}             return GREATEREQ;

{lexem}                 return yytext[0];
{float}                 yylval.val = strtol(yytext, NULL, 10); return FLOAT;
{identifier}            yylval.name = yytext; return IDENT;
{whitespace}+           ;
{comment}               ;
.                       { fprintf(stderr, "ERROR line %d: '%s'\n", yylloc.first_line, yytext); exit(1); }
