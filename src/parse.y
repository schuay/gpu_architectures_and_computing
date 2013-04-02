%{
    #include <stdio.h>
    #include <stdlib.h>

    #include "astnode.hpp"

    #define YYDEBUG 1

    int yylex();
    void yyerror(const char *p);

    int errcount = 0;
%}

%union {
    double val;
    const char *name;
    ASTNode *node;
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
                    { $<node>1->print(0); }
            ;
phi_expr    :   pred_expr
            |   unary_op phi_expr
                    { $<node>$ = new UnaryOpNode($<name>1, $<node>2); }
            |   '(' phi_expr ')' binary_op '(' phi_expr ')'
                    { $<node>$ = new BinaryOpNode($<name>4, $<node>2, $<node>6); }
            ;
pred_expr   :   mat_expr comp_op mat_expr
                    { $<node>$ = new BinaryOpNode($<name>2, $<node>1, $<node>3); }
            ;   /* Other pred_expr forms left out. */
mat_expr    :   IDENT
                    { $<node>$ = new IdentNode("ident", $<name>1); }
            |   FLOAT
                    { $<node>$ = new FloatNode("float", $<val>1); }
            ;   /* We don't support any arithmetic at the moment. */
comp_op     :   '<'
                    { $<name>$ = "<"; }
            |   '>'
                    { $<name>$ = ">"; }
            |   LESSEQ
                    { $<name>$ = "<="; }
            |   GREATEREQ
                    { $<name>$ = ">="; }
            ;
unary_op    :   NOT
                    { $<name>$ = "not"; }
            |   EV '_' '[' FLOAT ',' FLOAT ']'
                    { $<name>$ = "ev"; }
            |   EV
                    { $<name>$ = "ev"; }
            |   ALW '_' '[' FLOAT ',' FLOAT ']'
                    { $<name>$ = "alw"; }
            |   ALW
                    { $<name>$ = "alw"; }
            ;
binary_op   :   OR
                    { $<name>$ = "or"; }
            |   AND
                    { $<name>$ = "and"; }
            |   UNTIL '_' '[' FLOAT ',' FLOAT ']'
                    { $<name>$ = "until"; }
            |   UNTIL
                    { $<name>$ = "until"; }
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
