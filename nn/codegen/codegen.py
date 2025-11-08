import json
import warnings
from enum import Enum
import jinja2
from lark import Tree, Token

from cfl_grammar import parse


class ProcedureParser:
    def __init__(self, source: list, ctx_forward: bool):
        self.source = source
        self.ctx_forward = ctx_forward
        self.temp_identifiers = []

    def __convert_identifier(self, ast: Tree | Token) -> str:
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
                    if self.ctx_forward:
                        return f"result->shape_.{name}"
                    else:
                        return f"tensor_->shape_.{name}"
                elif shape in ("reduction", "pooling") and name == "$mask":
                    ret = "mask\\_"
                elif shape == "matmul" and name in ("$m", "$n", "$k"):
                    ret = f"matmul_{name[1:]}\\_"
                elif shape == "conv" and name in ("$n", "$ci", "$co", "$hi", "$wi", "$hk", "$wk", "$ho", "$wo"):
                    if name in ("$n", "$ci", "$co"):
                        ret = f"conv_{name[1:]}\\_"
                    else:
                        ret = {
                            "$hi": "conv_input\\_->shape_.lengths[1]",
                            "$wi": "conv_input\\_->shape_.lengths[0]",
                            "$hk": "conv_kernel\\_->shape_.lengths[1]",
                            "$wk": "conv_kernel\\_->shape_.lengths[0]",
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
                        ret = f"{first}\\_->shape_.{second}"
                    elif second == "grad" and not self.ctx_forward:
                        return f"{first}_->grad_data_"
                    elif shape == "broadcast" and second in {"mask", "shape_identity"}:
                        ret = f"{first}_{second}\\_"
                    elif shape == "matmul" and second == "transpose":
                        ret = f"transpose_{first}\\_"
                    else:
                        ret = None
                elif first == "result" and second in ("size", "ndim", "lengths"):
                    if self.ctx_forward:
                        return f"result->shape_.{second}"
                    else:
                        return f"tensor_->shape_.{second}"
                else:
                    ret = None

                if ret is None:
                    warnings.warn(f"unrecognized identifier {first}.{second} in operator {operator_name}")
                    return f"{first}.{second}"
                else:
                    return ret.replace("\\_", "" if self.ctx_forward else "_")

            case Tree("index", [first, Token("SIZE", index)]):
                return f"{self.__convert_identifier(first)}[{index}]"

            case _:
                raise RuntimeError(f"unexpected identifier {ast}")

    def __convert_expr(self, ast: Tree | Token) -> dict:
        match ast:
            case Token("NAME", _):
                return {"rule": "CONVERTED", "value": self.__convert_identifier(ast)}
            case Tree("attr", _):
                return {"rule": "CONVERTED", "value": self.__convert_identifier(ast)}
            case Tree("index", _):
                return {"rule": "CONVERTED", "value": self.__convert_identifier(ast)}
            case Token(data, value):
                return {"rule": data, "value": value}
            case Tree(data, children):
                return {"rule": data, "children": [self.__convert_expr(i) for i in children]}

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
                    case Tree("apply", [Token("NAME", grad_value)]):
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
                            "type": typename,
                            "size_expr": self.__convert_expr(expr)  # callback render
                        }
                        self.temp_identifiers.append(name)
                    case "apply":
                        if self.ctx_forward:
                            raise RuntimeError(f"propagating grad in forward context of operator {operator_name}")
                        item = {
                            "statement": "apply",
                            "grad_value": grad_value
                        }
                    case "value":
                        if size:
                            code = f"{typename} {name} [{size}] = {raw_value}"
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
                            "args": [self.__convert_expr(i) for i in args]  # callback render
                        }
                        if template_args:
                            item["template_args"] = [self.__convert_expr(i) for i in template_args]  # callback render

                item["indent"] = code_indent
                item["raw"] = x
                result.append(item)

            elif isinstance(x, dict):
                if "if" in x:
                    condition = parse(x["if"], is_expr=True)
                    result.append({
                        "control": "if",
                        "expr": self.__convert_expr(condition),  # callback render
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
                        result += self.__parse_procedure(x["then"], code_indent=code_indent + 1)
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


def render_expr(ast: Tree | Token, dtype=None, parent_precedence=0) -> str:
    # callback by jinja template
    pass


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

        shape_desc = operator["shape"]

        if "ewise" in shape_desc:
            shape = "ewise"
            tensors += shape_desc["ewise"]
            ewise_args = shape_desc["ewise"]
            shape_args = {
                "ewise_args": ewise_args
            }
        elif "identity" in shape_desc:
            shape = "identity"
            tensors.append(shape_desc["identity"])
            shape_args = {}
        elif "broadcast" in shape_desc:
            shape = "broadcast"
            tensors += shape_desc["broadcast"]
            broadcast_args = shape_desc["broadcast"]
            shape_args = {
                "broadcast_args": broadcast_args
            }
        elif "reduction" in shape_desc:
            shape = "reduction"
            tensors.append(shape_desc["reduction"]["source"])
            reduction_source = shape_desc["reduction"]["source"]
            reduction_dims = shape_desc["reduction"]["dims"]
            shape_args = {
                "reduction_source": reduction_source,
                "reduction_dims": reduction_dims
            }
        elif "pooling" in shape_desc:
            shape = "pooling"
            tensors.append(shape_desc["pooling"]["source"])
            pooling_source = shape_desc["pooling"]["source"]
            pooling_strides = shape_desc["pooling"]["strides"]
            for dim, stride in pooling_strides.items():
                sizes.append(stride)
            shape_args = {
                "pooling_source": pooling_source,
                "pooling_strides": pooling_strides
            }
        elif "matmul" in shape_desc:
            shape = "matmul"
            matmul_first = shape_desc["matmul"]["first"]
            matmul_second = shape_desc["matmul"]["second"]
            tensors += [matmul_first, matmul_second]
            shape_args = {
                "matmul_first": matmul_first,
                "matmul_second": matmul_second
            }
        elif "conv" in shape_desc:
            shape = "conv"
            conv_input = shape_desc["conv"]["input"]
            conv_kernel = shape_desc["conv"]["kernel"]
            conv_bias = shape_desc["conv"]["bias"]
            conv_h_padding = shape_desc["conv"]["h_padding"]
            conv_w_padding = shape_desc["conv"]["w_padding"]
            tensors += [conv_input, conv_kernel, conv_bias]
            sizes += [conv_h_padding, conv_w_padding]
            shape_args = {
                "conv_input": conv_input,
                "conv_kernel": conv_kernel,
                "conv_bias": conv_bias,
                "conv_h_padding": conv_h_padding,
                "conv_w_padding": conv_w_padding
            }
        else:
            raise RuntimeError("illegal shape")

        if "scalar" in shape_desc:
            scalars += shape_desc["scalar"]

        # procedure
        forward = ProcedureParser(operator["forward"], ctx_forward=True).run()

        if not operator["backward"] or operator["backward"] == "reject":
            backward = "reject"
        else:
            backward = {}
            for key, value in operator["backward"].items():
                backward[key] = ProcedureParser(value, ctx_forward=False).run()

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

        # forward jinja context
        if not scalars:
            base_ctx = {
                "tensors": tensors,
                "shape": shape,
                **shape_args,
                "fixed_dtype": False,
                "dtypes": dtypes_supported,
                "dtypes_unsupported": dtypes_unsupported,
                "procedure": forward,
                "allow_grad": (backward != "reject")
            }
            if backward != "reject":
                base_ctx.update({
                    "internal_name": operator_name,
                    "grad_args": ", ".join(["result"] + tensors + sizes)
                })

            for interface_type, interface in operator["interface"].items():
                if interface_type == "operator":
                    ctx = {
                        "name": f"operator{interface}",
                        "args": ", ".join(
                            [f"const tensor &{i}" for i in tensors[1:]] +
                            [f"size_t {i}" for i in sizes]
                        ),
                        "this": tensors[0],
                        **base_ctx
                    }
                    interface_ctx.append(ctx)
                elif interface_type == "inplace_operator":
                    pass  # todo
                elif interface_type == "function":
                    ctx = {
                        "name": interface,
                        "args": ", ".join(
                            [f"const tensor &{i}" for i in tensors] +
                            [f"size_t {i}" for i in sizes]
                        ),
                        **base_ctx
                    }
                    interface_ctx.append(ctx)
                else:
                    raise RuntimeError(f"illegal interface type {interface_type}")
        else:
            for dtype in dtypes_supported:
                base_ctx = {
                    "tensors": tensors,
                    "shape": shape,
                    **shape_args,
                    "fixed_dtype": dtype,
                    "procedure": forward,
                    "allow_grad": (backward != "reject")
                }
                if backward != "reject":
                    base_ctx.update({
                        "internal_name": operator_name,
                        "grad_args": ", ".join(["result"] + tensors + scalars + sizes)
                    })

                for interface_type, interface in operator["interface"].items():
                    if interface_type == "operator":
                        ctx = {
                            "name": f"operator{interface}",
                            "args": ", ".join(
                                [f"const tensor &{i}" for i in tensors[1:]] +
                                [f"{dtypes[dtype]["cpp_key"]} {i}" for i in scalars] +
                                [f"size_t {i}" for i in sizes]
                            ),
                            "this": tensors[0],
                            **base_ctx
                        }
                        interface_ctx.append(ctx)
                    elif interface_type == "inplace_operator":
                        pass  # todo
                    elif interface_type == "function":
                        ctx = {
                            "name": interface,
                            "args": ", ".join(
                                [f"const tensor &{i}" for i in tensors] +
                                [f"{dtypes[dtype]["cpp_key"]} {i}" for i in scalars] +
                                [f"size_t {i}" for i in sizes]
                            ),
                            **base_ctx
                        }
                        interface_ctx.append(ctx)
                    else:
                        raise RuntimeError(f"illegal interface type {interface_type}")

    with open("interface.generated.json", "w", encoding="utf-8") as f:
        json.dump(interface_ctx, f, indent=2, ensure_ascii=False)
