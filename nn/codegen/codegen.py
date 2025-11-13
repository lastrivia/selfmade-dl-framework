import json
import warnings
import jinja2
from lark import Tree, Token

from cfl_grammar import parse


class ProcedureParser:
    def __init__(self, source: list, ctx_forward: bool, types: list):
        self.source = source
        self.ctx_forward = ctx_forward
        self.temp_identifiers = []  # [*extern_symbols] if extern_symbols else []
        self.types = types

    def __render_identifier(self, ast: Tree | Token, has_array_index: bool = False) -> str:
        # resolve shape & name dependencies of alias names (NAME, NAME.NAME, NAME[SIZE])
        match ast:
            # using internal string @ret containing "\\_", for '_' suffices only in backward context
            case Token("NAME", name):
                if name in tensors:
                    ret = f"{name}\\_->data_"
                elif name == "result":
                    if self.ctx_forward:
                        return "result->data_"
                    else:
                        return "tensor_->data_"
                elif name == "grad" and not self.ctx_forward:
                    return "tensor_->grad_data_"
                elif name in ("size", "ndim", "lengths"):
                    element = "lengths.data()" if (name == "lengths" and not has_array_index) else name
                    if self.ctx_forward:
                        return f"result->shape_.{element}"
                    else:
                        return f"tensor_->shape_.{element}"
                elif shape in ("reduction", "pooling") and name == "$mask":
                    return "mask"
                elif shape == "matmul" and name in ("$m", "$n", "$k"):
                    ret = f"matmul_{name[1:]}\\_"
                elif shape == "conv" and name in ("$n", "$ci", "$co", "$hi", "$wi", "$hk", "$wk", "$ho", "$wo"):
                    if name in ("$n", "$ci", "$co"):
                        ret = f"conv_{name[1:]}\\_"
                    else:
                        ret = {
                            "$hi": "input\\_->shape_.lengths[1]",
                            "$wi": "input\\_->shape_.lengths[0]",
                            "$hk": "kernel\\_->shape_.lengths[1]",
                            "$wk": "kernel\\_->shape_.lengths[0]",
                            "$ho": "conv_h_out\\_",
                            "$wo": "conv_w_out\\_"
                        }[name]
                elif name in scalars or name in sizes:
                    ret = f"{name}\\_"
                elif name in self.temp_identifiers:
                    return name
                else:
                    warnings.warn(f"unrecognized identifier {name} in operator {operator_name}")
                    return name

                return ret.replace("\\_", "" if self.ctx_forward else "_")

            case Tree("attr", [Token("NAME", first), Token("NAME", second)]):
                if first in tensors:
                    if second in ("size", "ndim", "lengths"):
                        element = "lengths.data()" if (second == "lengths" and not has_array_index) else second
                        ret = f"{first}\\_->shape_.{element}"
                    elif second == "grad" and not self.ctx_forward:
                        return f"{first}_->grad_data_"
                    elif shape == "broadcast" and second == "mask":
                        return f"{first}_mask"
                    elif shape == "matmul" and second == "transpose":
                        ret = f"transpose_{first}\\_"
                    else:
                        ret = None
                elif first == "result" and second in ("size", "ndim", "lengths"):
                    element = "lengths.data()" if (second == "lengths" and not has_array_index) else second
                    if self.ctx_forward:
                        return f"result->shape_.{element}"
                    else:
                        return f"tensor_->shape_.{element}"
                else:
                    ret = None

                if ret is None:
                    warnings.warn(f"unrecognized identifier {first}.{second} in operator {operator_name}")
                    return f"{first}.{second}"
                else:
                    return ret.replace("\\_", "" if self.ctx_forward else "_")

            case Tree("index", [first, Token("SIZE", index)]):
                return f"{self.__render_identifier(first, has_array_index=True)}[{index}]"

            case _:
                raise RuntimeError(f"unexpected identifier {ast} in operator {operator_name}")

    def __render_expr(self, ast: Tree | Token) -> tuple[str, int]:

        match ast:
            # identifiers / values
            case Token("NAME", _):
                return self.__render_identifier(ast), 4
            case Tree("attr", _):
                return self.__render_identifier(ast), 4
            case Tree("index", _):
                return self.__render_identifier(ast), 4
            case Token("__CONTROL__", value):
                return value, 4
            case Token(_, value):
                return value, 4
            case Tree("auto", [Token("NUMBER", value)]):
                return f"{value}\\DTYPE_SUFFIX\\", 4

            # expression tree
            case Tree("comp", [left, Token("COMP_OP", op), right]):
                precedence = 0
                l_str, l_prec = self.__render_expr(left)
                r_str, r_prec = self.__render_expr(right)
                if l_prec < precedence:
                    l_str = f"({l_str})"
                if r_prec <= precedence:
                    r_str = f"({r_str})"
                return f"{l_str} {op} {r_str}", precedence
            case Tree(op, [left, right]) if op in ("add", "sub", "mul", "div"):
                precedence = {"add": 1, "sub": 1, "mul": 2, "div": 2}[op]
                l_str, l_prec = self.__render_expr(left)
                r_str, r_prec = self.__render_expr(right)
                if l_prec < precedence:
                    l_str = f"({l_str})"
                if r_prec <= precedence:
                    r_str = f"({r_str})"
                op_char = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
                return f"{l_str} {op_char} {r_str}", precedence
            case Tree(op, [child]) if op in ("not", "neg"):
                precedence = 3
                child_str, _ = self.__render_expr(child)
                op_char = {"not": "!", "neg": "-"}[op]
                return f"({op_char}{child_str})", precedence

            case _:
                raise RuntimeError(f"unexpected expr {ast} in operator {operator_name}")

    def __convert_expr(self, ast: Tree | Token) -> str | dict:
        expr, _ = self.__render_expr(ast)
        if "\\DTYPE_SUFFIX\\" in expr or "\\DTYPE_SIZE\\" in expr:
            ret = {}
            for dtype in self.types:
                suffix = dtypes[dtype]["suffix"]
                cpp_key = dtypes[dtype]["cpp_key"]
                ret[dtype] = (expr
                              .replace("\\DTYPE_SUFFIX\\", suffix)
                              .replace("\\DTYPE_SIZE\\", f"sizeof({cpp_key})"))
            return ret
        else:
            return expr

    # noinspection PyUnboundLocalVariable
    def __parse_procedure(self, source: list, code_indent=0) -> list:
        result = []
        for x in source:
            if isinstance(x, str):
                try:
                    ast = parse(x)
                except Exception as e:
                    raise RuntimeError(f"failed to parse {x} in procedure of operator {operator_name}: {e}")
                statement = ""
                match ast:
                    case Tree("workspace", [Token("NAME", name),
                                            expr,
                                            Token("TYPE", typename)]):
                        statement = "workspace"
                    case Tree("workspace", [Token("NAME", name),
                                            expr]):
                        statement = "workspace"
                        typename = "auto"
                    case Tree("apply", [grad_expr]):
                        statement = "apply"
                    case Tree("value", [Token("TYPE", typename),
                                        Token("NAME", name),
                                        Token("SIZE", size),
                                        Token("RAW", raw_value)]):
                        statement = "value"
                    case Tree("value", [Token("TYPE", typename),
                                        Token("NAME", name),
                                        Token("RAW", raw_value)]):
                        statement = "value"
                        size = None
                    case Tree("call", [Token("NAME", function),
                                       Tree("template_args", template_args),
                                       Tree("args", args)]):
                        statement = "call"
                    case Tree("call", [Token("NAME", function),
                                       Tree("args", args)]):
                        statement = "call"
                        template_args = []
                    case _:
                        raise RuntimeError(
                            f"illegal statement \"{x}\" => {ast} in procedure of operator {operator_name}")

                match statement:
                    case "workspace":
                        item = {
                            "statement": "workspace",
                            "name": name,
                            "size": self.__convert_expr(
                                Tree("mul", [expr, Token("__CONTROL__", "\\DTYPE_SIZE\\")])
                            )
                        }
                        self.temp_identifiers.append(name)
                    case "apply":
                        if self.ctx_forward:
                            raise RuntimeError(f"propagating grad in forward context of operator {operator_name}")
                        item = {
                            "statement": "apply",
                            "grad": self.__convert_expr(grad_expr)
                        }
                    case "value":
                        if size:
                            code = f"{typename} {name}[{size}] = {raw_value}"
                        else:
                            code = f"{typename} {name} = {raw_value}"
                        item = {
                            "statement": "value",
                            "code": code
                        }
                        self.temp_identifiers.append(name)
                    case "call":
                        item = {
                            "kernel": function,
                            "args": [self.__convert_expr(i) for i in args]
                        }
                        if template_args:
                            item["template_args"] = [self.__convert_expr(i) for i in template_args]

                item["indent"] = code_indent
                item["raw"] = x
                result.append(item)

            elif isinstance(x, dict):
                if "if" in x:
                    condition = parse(x["if"], is_expr=True)
                    result.append({
                        "control": "if",
                        "condition": self.__convert_expr(condition),
                        "indent": code_indent
                    })
                    result += self.__parse_procedure(x["then"], code_indent=code_indent + 1)
                    result.append({
                        "control": "end",
                        "indent": code_indent
                    })
                    if "else" in x:
                        result.append({
                            "control": "else",
                            "indent": code_indent
                        })
                        result += self.__parse_procedure(x["else"], code_indent=code_indent + 1)
                        result.append({
                            "control": "end",
                            "indent": code_indent
                        })
                else:
                    raise RuntimeError(f"illegal control procedure {x} in operator {operator_name}")
            else:
                raise RuntimeError(f"illegal procedure {x} in operator {operator_name}")
        return result

    def run(self) -> list:
        self.temp_identifiers.clear()
        return self.__parse_procedure(self.source)


if __name__ == "__main__":

    # load config
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    dtypes = config["dtypes"]
    operators = config["operators"]

    interface_ctx = []
    autograd_ctx = []

    for operator in operators:

        operator_name = operator["name"]

        # shape & args
        tensors = []
        scalars = []
        sizes = []

        other_interface_args = []

        saved_values = []
        saved_resources = []  # transfer ownership from interface to grad node

        shape_desc = operator["shape"]

        if "ewise" in shape_desc:
            shape = "ewise"
            tensors += shape_desc["ewise"]
            ewise_args = shape_desc["ewise"]
            shape_template_args = {
                "ewise_args": ewise_args
            }
        elif "identity" in shape_desc:
            shape = "identity"
            tensors.append(shape_desc["identity"])
            shape_template_args = {}
        elif "broadcast" in shape_desc:
            shape = "broadcast"
            tensors += shape_desc["broadcast"]
            broadcast_args = shape_desc["broadcast"]
            shape_template_args = {
                "broadcast_args": broadcast_args
            }
            saved_resources += [
                {"type": "workspace", "name": f"{i}_mask_workspace"} for i in broadcast_args
            ]
        elif "reduction" in shape_desc:
            shape = "reduction"
            tensors.append(shape_desc["reduction"]["source"])
            reduction_source = shape_desc["reduction"]["source"]
            reduction_dims = shape_desc["reduction"]["dims"]
            shape_template_args = {
                "reduction_source": reduction_source,
                "reduction_dims": reduction_dims
            }
            saved_resources += [
                {"type": "workspace", "name": "mask_workspace"}
            ]
            other_interface_args += [
                {"type": "std::vector<size_t>", "name": "dims"}
            ]
        elif "pooling" in shape_desc:
            shape = "pooling"
            tensors.append(shape_desc["pooling"]["source"])
            pooling_source = shape_desc["pooling"]["source"]
            pooling_strides = shape_desc["pooling"]["strides"]
            for dim, stride in pooling_strides.items():
                sizes.append(stride)
            shape_template_args = {
                "pooling_source": pooling_source,
                "pooling_strides": pooling_strides
            }
            saved_resources += [
                {"type": "workspace", "name": "mask_workspace"}
            ]
        elif "matmul" in shape_desc:
            shape = "matmul"
            matmul_first = shape_desc["matmul"]["first"]
            matmul_second = shape_desc["matmul"]["second"]
            tensors += [matmul_first, matmul_second]
            shape_template_args = {
                "matmul_first": matmul_first,
                "matmul_second": matmul_second
            }
            saved_values += [
                {"type": "bool", "name": f"transpose_{matmul_first}"},
                {"type": "bool", "name": f"transpose_{matmul_second}"},
                {"type": "size_t", "name": "matmul_m"},
                {"type": "size_t", "name": "matmul_n"},
                {"type": "size_t", "name": "matmul_k"}
            ]
        elif "conv" in shape_desc:
            shape = "conv"
            conv_input = shape_desc["conv"]["input"]
            conv_kernel = shape_desc["conv"]["kernel"]
            conv_bias = shape_desc["conv"]["bias"]
            conv_h_padding = shape_desc["conv"]["h_padding"]
            conv_w_padding = shape_desc["conv"]["w_padding"]
            tensors += [conv_input, conv_kernel, conv_bias]
            sizes += [conv_h_padding, conv_w_padding]
            shape_template_args = {
                "conv_input": conv_input,
                "conv_kernel": conv_kernel,
                "conv_bias": conv_bias,
                "conv_h_padding": conv_h_padding,
                "conv_w_padding": conv_w_padding
            }
            saved_values += [
                {"type": "size_t", "name": "conv_n"},
                {"type": "size_t", "name": "conv_ci"},
                {"type": "size_t", "name": "conv_co"},
                {"type": "size_t", "name": "conv_h_out"},
                {"type": "size_t", "name": "conv_w_out"}
            ]
        else:
            raise RuntimeError("illegal shape")

        if "scalar" in shape_desc:
            scalars += shape_desc["scalar"]

        # dtypes
        dtypes_supported = []
        dtypes_unsupported = []
        for dtype in operator["dtypes"]:
            if dtype not in dtypes:
                warnings.warn(f"unknown dtype {dtype} in operator {operator_name}")
        for dtype in dtypes:
            if dtype in operator["dtypes"]:
                dtypes_supported.append(dtype)
            else:
                dtypes_unsupported.append(dtype)

        # procedure
        forward = ProcedureParser(operator["forward"], ctx_forward=True, types=dtypes_supported).run()

        if not operator["backward"] or operator["backward"] == "reject":
            backward = "reject"
        else:
            # if "transfer_workspaces" in operator:
            #     extern_symbols = operator["transfer_workspaces"]
            #     for i in extern_symbols:
            #         saved_resources.append({"type": "workspace", "name": i})
            # else:
            #     extern_symbols = []
            backward = {}
            for key, value in operator["backward"].items():
                backward[key] = ProcedureParser(value, ctx_forward=False, types=dtypes_supported).run()

        # forward jinja context
        ctx_base = {
            "tensors": tensors,
            "shape": shape,
            **shape_template_args,
            "procedure": forward,
            "allow_grad": (backward != "reject")
        }
        if backward != "reject":
            ctx_base.update({
                "internal_name": operator_name,
                "grad_args": ", ".join(
                    ["result"] +
                    tensors + scalars + sizes +
                    [i["name"] for i in saved_values] +
                    [f"std::move({i["name"]})" for i in saved_resources]
                )
            })

        if not scalars:
            # generate branches for each type inside the interface
            ctx_type = {
                "fixed_dtype": False,
                "dtypes": dtypes_supported,
                "dtypes_unsupported": dtypes_unsupported
            }

            for interface_type, interface in operator["interface"].items():
                if interface_type == "operator":
                    ctx = {
                        "name": f"operator{interface}",
                        "args": ", ".join(
                            [f"const tensor &{i}" for i in tensors[1:]] +
                            [f"size_t {i}" for i in sizes] +
                            [f"{i['type']} {i['name']}" for i in other_interface_args]
                        ),
                        "this": tensors[0],
                        **ctx_type,
                        **ctx_base
                    }
                    interface_ctx.append(ctx)
                elif interface_type == "inplace_operator":
                    pass  # todo
                elif interface_type == "function":
                    ctx = {
                        "name": interface,
                        "args": ", ".join(
                            [f"const tensor &{i}" for i in tensors] +
                            [f"size_t {i}" for i in sizes] +
                            [f"{i['type']} {i['name']}" for i in other_interface_args]
                        ),
                        **ctx_type,
                        **ctx_base
                    }
                    interface_ctx.append(ctx)
                else:
                    raise RuntimeError(f"illegal interface type {interface_type}")
        else:
            # generate overloaded interfaces for each type
            for dtype in dtypes_supported:
                dtype_cpp_key = dtypes[dtype]["cpp_key"]
                ctx_type = {
                    "fixed_dtype": dtype
                }

                for interface_type, interface in operator["interface"].items():
                    if interface_type == "operator":
                        ctx = {
                            "name": f"operator{interface}",
                            "args": ", ".join(
                                [f"const tensor &{i}" for i in tensors[1:]] +
                                [f"{dtype_cpp_key} {i}" for i in scalars] +
                                [f"size_t {i}" for i in sizes] +
                                [f"{i['type']} {i['name']}" for i in other_interface_args]
                            ),
                            "this": tensors[0],
                            **ctx_type,
                            **ctx_base
                        }
                        interface_ctx.append(ctx)
                    elif interface_type == "inplace_operator":
                        pass  # todo
                    elif interface_type == "function":
                        ctx = {
                            "name": interface,
                            "args": ", ".join(
                                [f"const tensor &{i}" for i in tensors] +
                                [f"{dtype_cpp_key} {i}" for i in scalars] +
                                [f"size_t {i}" for i in sizes] +
                                [f"{i['type']} {i['name']}" for i in other_interface_args]
                            ),
                            **ctx_type,
                            **ctx_base
                        }
                        interface_ctx.append(ctx)
                    else:
                        raise RuntimeError(f"illegal interface type {interface_type}")

        # backward jinja context
        if backward != "reject":
            saved_members = {}
            for i in saved_values + saved_resources:
                if i["type"] in saved_members:
                    saved_members[i["type"]].append(i["name"])
                else:
                    saved_members[i["type"]] = [i["name"]]

            for dtype in dtypes_supported:
                dtype_cpp_key = dtypes[dtype]["cpp_key"]
                ctx = {
                    "internal_name": operator_name,
                    "dtype": dtype,
                    "shape": shape,
                    **shape_template_args,
                    "tensors": tensors,
                    "init_args": ", ".join(
                        ["const tensor &result"] +
                        [f"const tensor &{i}" for i in tensors] +
                        [f"{dtype_cpp_key} {i}" for i in scalars] +
                        [f"size_t {i}" for i in sizes] +
                        [f"{i["type"]} {i["name"]}" for i in saved_values] +
                        [f"{i["type"]} &&{i["name"]}" for i in saved_resources]
                    ),
                    "init_proc": ", ".join(
                        ["grad_node(result)"] +
                        [f"{i}_({i})" for i in (tensors + scalars + sizes)] +
                        [f"{i["name"]}_({i["name"]})" for i in saved_values] +
                        [f"{i["name"]}_(std::move({i["name"]}))" for i in saved_resources] +
                        [f"{i}_ver_({i}->version_)" for i in tensors]
                    ),
                    "members": [
                        "tensor " + ", ".join([f"{i}_" for i in tensors]),
                        *([f"{dtype_cpp_key} " + ", ".join([f"{i}_" for i in scalars])] if scalars else []),
                        *(["size_t " + ", ".join([f"{i}_" for i in sizes])] if sizes else []),
                        *[
                            f"{t} " + ", ".join([f"{i}_" for i in names])
                            for t, names in saved_members.items()
                        ],
                        "size_t " + ", ".join([f"{i}_ver_" for i in tensors])
                    ],
                    "procedure": backward
                }
                autograd_ctx.append(ctx)

    with open("generated/interface_ctx.generated.json", "w", encoding="utf-8") as f:
        json.dump(interface_ctx, f, indent=2, ensure_ascii=False)
    with open("generated/autograd_ctx.generated.json", "w", encoding="utf-8") as f:
        json.dump(autograd_ctx, f, indent=2, ensure_ascii=False)

    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates"))

    interface_decl_template = jinja_env.get_template("interface_decl.h.j2")
    interface_impl_template = jinja_env.get_template("interface_impl.h.j2")
    autograd_decl_template = jinja_env.get_template("autograd_decl.h.j2")
    autograd_impl_template = jinja_env.get_template("autograd_impl.h.j2")

    interface_decl = []
    interface_impl = [
        r'''#pragma once
        
#include "backend.h"
#include "tensor/tensor_impl.h"
#include "tensor/autograd.h"'''
    ]

    for ctx in interface_ctx:
        try:
            decl = interface_decl_template.render(**ctx)
        except Exception as e:
            name = ctx["internal_name"]
            raise RuntimeError(f"failed to render interface decl {name}: {e}\n ctx: {ctx}")
        interface_decl.append(decl)
        try:
            impl = interface_impl_template.render(**ctx)
        except Exception as e:
            name = ctx["internal_name"]
            raise RuntimeError(f"failed to render interface impl {name}: {e}\n ctx: {ctx}")
        interface_impl.append(impl)

    autograd_decl = []
    autograd_impl = [
        r'''#pragma once
        
#include "backend.h"
#include "tensor/tensor_impl.h"
#include "tensor/autograd_base.h"'''
    ]

    for ctx in autograd_ctx:
        try:
            decl = autograd_decl_template.render(**ctx)
        except Exception as e:
            name = ctx["internal_name"]
            raise RuntimeError(f"failed to render autograd decl {name}: {e}\n ctx: {ctx}")
        autograd_decl.append(decl)
        try:
            impl = autograd_impl_template.render(**ctx)
        except Exception as e:
            name = ctx["internal_name"]
            raise RuntimeError(f"failed to render autograd impl {name}: {e}\n ctx: {ctx}")
        autograd_impl.append(impl)

    with open("generated/interface_decl.generated.h", "w", encoding="utf-8") as f:
        f.write("\n".join(interface_decl))
    with open("generated/interface_impl.generated.h", "w", encoding="utf-8") as f:
        f.write("\n\n".join(interface_impl))
    with open("generated/autograd_decl.generated.h", "w", encoding="utf-8") as f:
        f.write("\n".join(autograd_decl))
    with open("generated/autograd_impl.generated.h", "w", encoding="utf-8") as f:
        f.write("\n\n".join(autograd_impl))
