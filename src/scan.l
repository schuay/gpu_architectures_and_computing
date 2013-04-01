%{
//    #define YY_USER_ACTION yylloc.first_line = yylloc.last_line = yylineno;

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
identifier_body     ({letter}|{digit})
identifier          {letter}{identifier_body}*
whitespace          [ \t\n]
lexem               [()<>,\]\[]

    int errors = 0;
    
    enum { EV, ALW, UNTIL, NOT, AND, OR, LESSEQ, GREATEREQ, IDENT };
%%

{eventually}            printf("EV"); //return EV;
{always}                printf("ALW"); //return ALW;
until                   printf("UNTIL"); //return UNTIL;
not                     printf("NOT"); //return NOT;
and                     printf("AND"); //return AND;
or                      printf("OR"); //return OR;
{lesseq}                printf("LESSEQ"); //return LESSEQ;
{greatereq}             printf("GREATEREQ"); //return GREATEREQ;

{lexem}                 printf("%c", yytext[0]); //return yytext[0];
{identifier}            printf("IDENT"); //return IDENT; // yylval.sym = syms.insert(yytext); return IDENT;
{whitespace}+           ;
{comment}               ;
.                       { fprintf(stderr, "ERROR line %d: '%s'\n", 0 /* yylloc.first_line */, yytext); exit(1); }

%%

int main(void) {
    yylex();
    if (errors) {
        fprintf(stderr, "%d error(s) occurred.\n", errors);
        return 1;
    }
    return 0;
}
