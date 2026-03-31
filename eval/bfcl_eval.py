"""BFCL (Berkeley Function Call Leaderboard) evaluation.

Ported from gorilla/berkeley-function-call-leaderboard/bfcl_eval/.
Supports: simple_python, multiple, parallel, parallel_multiple categories.
Python-only, stdlib only (re, ast, json).
"""

import ast
import json
import re


# ============================================================================
# Constants
# ============================================================================

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]
NESTED_CONVERSION_TYPE_LIST = ["Array", "ArrayList", "array"]


# ============================================================================
# Parsing: AST and JSON
# ============================================================================

def resolve_ast_by_type(value):
    """Convert AST node to Python value."""
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(value, ast.NameConstant):
        output = value.value
    elif isinstance(value, ast.BinOp):
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


def resolve_ast_call(elem):
    """Convert AST Call node to {func_name: {params}}."""
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def ast_parse(input_str: str) -> list:
    """Parse Python function calls from string using ast.parse."""
    cleaned_input = input_str.strip().strip("'")
    parsed = ast.parse(cleaned_input, mode="eval")
    extracted = []
    if isinstance(parsed.body, ast.Call):
        extracted.append(resolve_ast_call(parsed.body))
    else:
        for elem in parsed.body.elts:
            assert isinstance(elem, ast.Call)
            extracted.append(resolve_ast_call(elem))
    return extracted


def parse_json_function_call(source_code):
    """Parse JSON function calls from string."""
    json_match = re.search(r"\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\]", source_code, re.DOTALL)
    if json_match:
        source_code = json_match.group(0)

    try:
        json_dict = json.loads(source_code)
    except json.JSONDecodeError:
        return []

    function_calls = []
    for function_call in json_dict:
        if isinstance(function_call, dict):
            function_name = function_call["function"]
            arguments = function_call["parameters"]
            function_calls.append({function_name: arguments})
    return function_calls


def default_decode_ast_prompting(result: str) -> list:
    """Decode AST from model output: strip backticks, wrap in [], parse."""
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    return ast_parse(result)


def is_function_calling_format_output(decoded_output):
    """Validate output is list[dict] with shape [{func: {params}}]."""
    if type(decoded_output) != list:
        return False
    for item in decoded_output:
        if type(item) != dict:
            return False
        if len(item) != 1:
            return False
        if type(list(item.values())[0]) != dict:
            return False
    return True


# ============================================================================
# Validation: Type and Value Checking
# ============================================================================

def standardize_string(input_string: str):
    """Remove spaces, punctuation; lowercase; single→double quotes."""
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def get_possible_answer_type(possible_answer: list):
    """Infer type from first non-empty answer."""
    for answer in possible_answer:
        if answer != "":
            return type(answer)
    return None


def find_description(func_descriptions, name):
    """Lookup function schema by name."""
    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    else:
        return func_descriptions


def type_checker(
    param: str,
    value,
    possible_answer: list,
    expected_type_description: str,
    expected_type_converted,
    nested_type_converted,
):
    """Validate type of parameter value (Python-only, one level nesting)."""
    result = {
        "valid": True,
        "error": [],
        "is_variable": False,
        "error_type": "type_error:simple",
    }

    is_variable = False
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type != None:
        if possible_answer_type != expected_type_converted:
            is_variable = True

    if type(value) == expected_type_converted:
        if nested_type_converted == None:
            result["is_variable"] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True
                if type(possible_answer_item) == list:
                    for value_item in value:
                        checker_result = type_checker(
                            param,
                            value_item,
                            possible_answer_item,
                            str(nested_type_converted),
                            nested_type_converted,
                            None,
                        )
                        if not checker_result["valid"]:
                            flag = False
                            break

                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}

            result["valid"] = False
            result["error"] = [
                f"Nested type checking failed for parameter {repr(param)}. Expected outer type {expected_type_description} with inner type {str(nested_type_converted)}. Parameter value: {repr(value)}."
            ]
            result["error_type"] = "type_error:nested"

    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type != None:
        if type(value) == possible_answer_type:
            result["is_variable"] = True
            return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {repr(param)}. Expected type {expected_type_description}, got {type(value).__name__}. Parameter value: {repr(value)}."
    )
    result["error_type"] = "type_error:simple"
    return result


def string_checker(param: str, model_output: str, possible_answer: list):
    """Validate string parameter with normalization."""
    standardize_possible_answer = []
    standardize_model_output = standardize_string(model_output)
    for i in range(len(possible_answer)):
        if type(possible_answer[i]) == str:
            standardize_possible_answer.append(standardize_string(possible_answer[i]))

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}."
            ],
            "error_type": "value_error:string",
        }

    return {"valid": True, "error": []}


def list_checker(param: str, model_output: list, possible_answer: list):
    """Validate list parameter with string normalization."""
    model_output = [standardize_string(item) if type(item) == str else item for item in model_output]

    standardize_possible_answer = []
    for i in range(len(possible_answer)):
        standardize_possible_answer.append([])
        for j in range(len(possible_answer[i])):
            if type(possible_answer[i][j]) == str:
                standardize_possible_answer[i].append(standardize_string(possible_answer[i][j]))
            else:
                standardize_possible_answer[i].append(possible_answer[i][j])

    if model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}."
            ],
            "error_type": "value_error:list/tuple",
        }

    return {"valid": True, "error": []}


def dict_checker(param: str, model_output: dict, possible_answers: list):
    """Validate dict parameter."""
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for i in range(len(possible_answers)):

        if possible_answers[i] == "":
            continue

        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        possible_answer = possible_answers[i]

        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

            standardize_value = value
            if type(value) == str:
                standardize_value = standardize_string(value)

            standardize_possible_answer = []
            for j in range(len(possible_answer[key])):
                if type(possible_answer[key][j]) == str:
                    standardize_possible_answer.append(standardize_string(possible_answer[key][j]))
                else:
                    standardize_possible_answer.append(possible_answer[key][j])

            if standardize_value not in standardize_possible_answer:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {repr(key)}: {repr(value)}. Expected one of {standardize_possible_answer}."
                )
                result["error_type"] = "value_error:dict_value"
                flag = False
                break

        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}

    return result


def list_dict_checker(param: str, model_output: list, possible_answers: list):
    """Validate list-of-dicts parameter."""
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}

    for answer_index in range(len(possible_answers)):
        flag = True

        if len(model_output) != len(possible_answers[answer_index]):
            result["valid"] = False
            result["error"] = ["Wrong number of dictionaries in the list."]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue

        for dict_index in range(len(model_output)):
            result = dict_checker(
                param,
                model_output[dict_index],
                [possible_answers[answer_index][dict_index]],
            )
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}

    return result


# ============================================================================
# Function Call Checkers
# ============================================================================

def simple_function_checker(
    func_description: dict,
    model_output: dict,
    possible_answer: dict,
    underscore_to_dot: bool = False,
):
    """Validate single function call (Python-only)."""
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"]["required"]

    result = {
        "valid": True,
        "error": [],
        "error_type": "simple_function_checker:unclear",
    }

    if underscore_to_dot and "." in func_name:
        func_name = re.sub(r"\.", "_", func_name)

    if func_name not in model_output:
        result["valid"] = False
        result["error"].append(f"Function name {repr(func_name)} not found in model output.")
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_output[func_name]

    for param in required_params:
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            result["valid"] = False
            result["error"].append(f"Unexpected parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result

        full_param_details = param_details[param]
        expected_type_description = full_param_details["type"]
        is_variable = False
        nested_type_converted = None

        expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested_type = param_details[param]["items"]["type"]
            nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]

        if expected_type_description == "tuple" and type(value) == tuple:
            value = list(value)

        if expected_type_description == "float" and type(value) == int:
            value = float(value)

        type_check_result = type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            return type_check_result

        if not is_variable:
            if expected_type_converted == dict:
                result = dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            elif expected_type_converted == list and nested_type_converted == dict:
                result = list_dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            elif expected_type_converted == str:
                result = string_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            elif expected_type_converted == list:
                result = list_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

        if value not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Invalid value for parameter {repr(param)}: {repr(value)}. Expected one of {possible_answer[param]}."
            )
            result["error_type"] = "value_error:others"
            return result

    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(f"Optional parameter {repr(param)} not provided and not marked as optional.")
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def multiple_function_checker(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
    underscore_to_dot: bool = False,
):
    """Validate multiple function calls (all same function)."""
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "multiple_function_checker:wrong_count",
        }

    func_name_expected = list(possible_answers[0].keys())[0]
    func_description = find_description(func_descriptions, func_name_expected)
    return simple_function_checker(
        func_description,
        model_output[0],
        possible_answers[0],
        underscore_to_dot,
    )


def parallel_function_checker_no_order(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
    underscore_to_dot: bool = False,
):
    """Validate parallel function calls (unordered)."""
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }

    matched_indices = []

    for i in range(len(possible_answers)):
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = find_description(func_descriptions, func_name_expected)

        all_errors = []

        for index in range(len(model_output)):
            if index in matched_indices:
                continue

            result = simple_function_checker(
                func_description,
                model_output[index],
                possible_answers[i],
                underscore_to_dot,
            )

            if result["valid"]:
                matched_indices.append(index)
                break
            else:
                all_errors.append(
                    {
                        f"Model Result Index {index}": {
                            "sub_error": result["error"],
                            "sub_error_type": result["error_type"],
                            "model_output_item": model_output[index],
                            "possible_answer_item": possible_answers[i],
                        }
                    }
                )

        if not result["valid"]:
            considered_indices = [j for j in range(len(model_output)) if j not in matched_indices]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered_indices} of model output for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }

    return {"valid": True, "error": []}


def ast_checker(
    func_description,
    model_output,
    possible_answer,
    test_category: str,
    underscore_to_dot: bool = False,
):
    """Dispatch to appropriate checker based on test_category."""
    if "parallel" in test_category:
        return parallel_function_checker_no_order(
            func_description, model_output, possible_answer, underscore_to_dot
        )

    elif "multiple" in test_category:
        return multiple_function_checker(
            func_description, model_output, possible_answer, underscore_to_dot
        )

    else:
        if len(model_output) != 1:
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "simple_function_checker:wrong_count",
            }

        return simple_function_checker(
            func_description[0], model_output[0], possible_answer[0], underscore_to_dot
        )


# ============================================================================
# Top-level Evaluation
# ============================================================================

def evaluate_bfcl(trajectory: dict) -> dict:
    """Evaluate a BFCL trajectory.

    Args:
        trajectory: dict with keys:
            - generated_text: str (raw model output)
            - ground_truth: list[dict] (possible answers, structured)
            - func_description: list[dict] (function schemas)
            - test_category: str (simple_python|multiple|parallel|parallel_multiple)

    Returns:
        dict with keys: prediction, ground_truth, score, metric, extraction_failed
    """
    generated_text = trajectory.get("generated_text", "")
    ground_truth = trajectory.get("ground_truth", [])
    func_description = trajectory.get("func_description", [])
    test_category = trajectory.get("test_category", "simple_python")

    # Try to parse model output
    decoded = None
    try:
        decoded = default_decode_ast_prompting(generated_text)
    except Exception:
        pass

    if decoded is None:
        try:
            decoded = parse_json_function_call(generated_text)
        except Exception:
            pass

    if decoded is None or not decoded or not is_function_calling_format_output(decoded):
        return {
            "prediction": None,
            "ground_truth": ground_truth,
            "score": 0.0,
            "metric": "accuracy",
            "extraction_failed": True,
        }

    # Validate against ground truth
    result = ast_checker(func_description, decoded, ground_truth, test_category)

    score = 1.0 if result["valid"] else 0.0

    return {
        "prediction": decoded,
        "ground_truth": ground_truth,
        "score": score,
        "metric": "accuracy",
        "extraction_failed": False,
    }
