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
                    { UnaryOpNode *n = (UnaryOpNode *)$<node>1;
                      n->expr = $<node>2; $<node>$ = n; }
            |   '(' phi_expr ')' binary_op '(' phi_expr ')'
                    { BinaryOpNode *n = (BinaryOpNode *)$<node>4;
                      n->lhs = $<node>2; n->rhs = $<node>6; $<node>$ = n; }
            ;
pred_expr   :   mat_expr comp_op mat_expr
                    { BinaryOpNode *n = (BinaryOpNode *)$<node>2;
                      n->lhs = $<node>1; n->rhs = $<node>3; $<node>$ = n; }
            ;   /* Other pred_expr forms left out. */
mat_expr    :   IDENT
                    { $<node>$ = new IdentNode("ident", $<name>1); }
            |   FLOAT
                    { $<node>$ = new FloatNode("float", $<val>1); }
            ;   /* We don't support any arithmetic at the moment. */
comp_op     :   '<'
                    { $<node>$ = new BinaryOpNode("<"); }
            |   '>'
                    { $<node>$ = new BinaryOpNode(">"); }
            |   LESSEQ
                    { $<node>$ = new BinaryOpNode("<="); }
            |   GREATEREQ
                    { $<node>$ = new BinaryOpNode(">="); }
            ;
unary_op    :   NOT
                    { $<node>$ = new UnaryOpNode("not"); }
            |   EV '_' '[' FLOAT ',' FLOAT ']'
                    { $<node>$ = new UnaryOpNode("ev", $<val>4, $<val>6); }
            |   EV
                    { $<node>$ = new UnaryOpNode("ev"); }
            |   ALW '_' '[' FLOAT ',' FLOAT ']'
                    { $<node>$ = new UnaryOpNode("alw", $<val>4, $<val>6); }
            |   ALW
                    { $<node>$ = new UnaryOpNode("alw"); }
            ;
binary_op   :   OR
                    { $<node>$ = new BinaryOpNode("or"); }
            |   AND
                    { $<node>$ = new BinaryOpNode("and"); }
            |   UNTIL '_' '[' FLOAT ',' FLOAT ']'
                    { $<node>$ = new BinaryOpNode("until", $<val>4, $<val>6); }
            |   UNTIL
                    { $<node>$ = new BinaryOpNode("until"); }
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
