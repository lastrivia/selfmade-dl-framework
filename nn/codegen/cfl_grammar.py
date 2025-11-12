import json

import lark

expr_grammar = r"""
?expr: sum
     | sum COMP_OP sum        -> comp

COMP_OP: "<" | ">" | "<=" | ">=" | "==" | "!="

?sum: sum "+" product         -> add
    | sum "-" product         -> sub
    | product
?product: product "*" factor  -> mul
        | product "/" factor  -> div
        | factor
?factor: atom
       | "(" expr ")"
       | NUMBER
       | auto
       | "!" factor           -> not
       | "-" factor           -> neg
       | "+" factor
?atom: NAME
     | atom "." NAME          -> attr
     | atom "[" SIZE "]"      -> index
auto: "auto" "(" NUMBER ")"   -> auto

TYPE: /(float|int|bool|size_t)/
NAME: /[$A-Za-z_][$\w]*/
NUMBER: /-?\d+(\.\d+)?/
      | "true" | "false"
SIZE: /\d+/
RAW: /\S[^\n]*/                                      // rest of line as raw string

%ignore /\s+/
"""

statement_grammar = r"""
?start: statement

?statement: workspace_stmt
          | apply_stmt
          | value_stmt
          | call

workspace_stmt: "workspace" NAME expr TYPE?          -> workspace

apply_stmt: "apply" NAME                             -> apply

value_stmt: "value" TYPE NAME array_size? "=" RAW    -> value
?array_size: "[" SIZE "]"

call: NAME "(" [args] ")"                        -> call
    | NAME "<" template_args ">" "(" [args] ")"  -> call
template_args: expr ("," expr)*                  -> template_args
args: expr ("," expr)*                           -> args

""" + expr_grammar

single_expr_grammar = r"""
?start: expr
""" + expr_grammar


# def tree_to_dict(t):
#     if isinstance(t, lark.Tree):
#         return {t.data: [tree_to_dict(child) for child in t.children]}
#     elif isinstance(t, lark.Token):
#         return str(t)
#     else:
#         return t

expr_parser = lark.Lark(single_expr_grammar, parser='lalr')
statement_parser = lark.Lark(statement_grammar, parser='lalr')

def parse(source, is_expr=False):
    parser = expr_parser if is_expr else statement_parser
    tree = parser.parse(source)
    return tree


if __name__ == '__main__':
    test_string = [
        "workspace tmp t.size float",
        "workspace tmp t.size",
        "mul_scalar(t.size, tmp, tmp, auto(2))",
        "mul_scalar(size, result, t, -scalar)",
        "maxpool_backward(t.size / t.lengths[1] / t.lengths[0], t.lengths[1], t.lengths[0], h_stride, w_stride, tmp, $mask, t)",
        "gemm<false, !b.transpose>($m, $n, $k, tmp, grad, b)",
        "conv_input_grad($n, $ci, $co, grad, $ho, $wo, kernel, $hk, $wk, $hk - h_padding - 1, $wk - w_padding - 1, tmp)",
        "value bool sum_mask[4] = {false, false, true, false}",
        "apply tmp"
    ]
    # tree = parse(test_string[3])
    tree = parse("(a+b*7 == 5) == true", is_expr=True)
    print(tree.pretty())
    print(tree)
    # print(json.dumps(tree, indent=2))
