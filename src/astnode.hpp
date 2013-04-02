class ASTNode
{
public:
    ASTNode(const char *type) :
        type(type)
    { }

    virtual void print(int level) {
        printf("%*s%s\n", level * 4, "", type);
    }

    const char *type;
};

class UnaryOpNode : public ASTNode
{
public:
    UnaryOpNode(const char * type, ASTNode *expr) :
        ASTNode(type), expr(expr)
    { }

    void print(int level) {
        ASTNode::print(level);
        expr->print(level + 1);
    }

    ASTNode *expr;
};

class BinaryOpNode : public ASTNode
{
public:
    BinaryOpNode(const char * type, ASTNode *lhs, ASTNode *rhs) :
        ASTNode(type), lhs(lhs), rhs(rhs)
    { }

    void print(int level) {
        ASTNode::print(level);
        lhs->print(level + 1);
        rhs->print(level + 1);
    }

    ASTNode *lhs, *rhs;
};

class IdentNode : public ASTNode
{
public:
    IdentNode(const char * type, const char * ident) :
        ASTNode(type), ident(ident)
    { }

    const char *ident;
};

class FloatNode : public ASTNode
{
public:
    FloatNode(const char * type, float value) :
        ASTNode(type), value(value)
    { }

    float value;
};
