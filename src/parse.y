%{
    #include <stdio.h>
    #include <stdlib.h>

    #define YYDEBUG 1

    int yylex();
    void yyerror(const char *p);

    int errcount = 0;
%}

%union {
    double val;
    const char *name;
}

%locations

%start formula
%token IDENT FLOAT
%nonassoc LESSEQ GREATEREQ '<' '>'
%left EV ALW UNTIL AND NOT OR

%%

/* We will need to alter this grammar to be unique (forms such as
 * mat_expr comp_op mat_expr won't work). */

formula     :   phi_expr
            ;
phi_expr    :   pred_expr
            |   unary_op phi_expr
            |   '(' phi_expr ')' binary_op '(' phi_expr ')'
            ;
pred_expr   :   mat_expr comp_op mat_expr
            ;   /* Other pred_expr forms left out. */
mat_expr    :   IDENT
            |   FLOAT
            ;   /* We don't support any arithmetic at the moment. */
comp_op     :   '<'
            |   '>'
            |   LESSEQ
            |   GREATEREQ
            ;
unary_op    :   NOT
            |   EV '_' '[' FLOAT ',' FLOAT ']'
            |   EV
            |   ALW '_' '[' FLOAT ',' FLOAT ']'
            |   ALW
            ;
binary_op   :   OR
            |   AND
            |   UNTIL '_' '[' FLOAT ',' FLOAT ']'
            |   UNTIL
            ;

%%

void yyerror(const char *p) {
    fprintf(stderr, "ERROR line %d: %s\n", yylloc.first_line, p);
    errcount++;
}

int main(void) {
    yydebug = 0;
    yyparse();
    if (errcount > 0) {
        return 2;
    }
    return 0;
}
