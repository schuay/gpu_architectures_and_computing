%{
    #include "parse.tab.h"

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
lexem               [()<>,\]\[]

    int errors = 0;
%%

{eventually}            printf("EV"); return EV;
{always}                printf("ALW"); return ALW;
until                   printf("UNTIL"); return UNTIL;
not                     printf("NOT"); return NOT;
and                     printf("AND"); return AND;
or                      printf("OR"); return OR;
{lesseq}                printf("LESSEQ"); return LESSEQ;
{greatereq}             printf("GREATEREQ"); return GREATEREQ;

{lexem}                 printf("%c", yytext[0]); return yytext[0];
{float}                 printf("%s", yytext); yylval.val = strtol(yytext, NULL, 10); return FLOAT;
{identifier}            printf("IDENT"); yylval.name = yytext; return IDENT;
{whitespace}+           ;
{comment}               ;
.                       { fprintf(stderr, "ERROR line %d: '%s'\n", yylloc.first_line, yytext); exit(1); }
