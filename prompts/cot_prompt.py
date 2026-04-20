"""
Zero-shot Chain-of-Thought prompts for each supported dataset.

Each prompt instructs the model to reason through the problem step-by-step
using 'Step i:' markers, then emit a clearly delimited final answer.
"""

# ---------------------------------------------------------------------------
# Shared system instruction (injected as the system turn for all datasets)
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are an expert reasoning assistant. "
    "For every problem you receive, think carefully and reason step-by-step. "
    "Label each reasoning step as 'Step 1:', 'Step 2:', etc. "
    "After all steps, write your final answer on a new line in "
    "\\boxed{your answer} format. You must double-escape all LaTeX backslashes. For example, output \\boxed instead of \boxed."
    "Keep your reasoning concise, and be brief in each step."
)

SYSTEM_QWEN_THINKING = (
    "You are an expert reasoning assistant. "
    "For every problem you receive, think carefully and reason step-by-step between <think> and </think>. "
    "Label each reasoning step between <think> and </think> as 'Step 1:', 'Step 2:', etc. "
    "After all steps, write your final answer on a new line in "
    "\\boxed{your answer} format. You must double-escape all LaTeX backslashes. For example, output \\boxed instead of \boxed."
    "Keep your reasoning concise, and be brief in each step."
)




SYSTEM_WITHOUT_BOXED = (
    "You are an expert reasoning assistant. "
    "For every problem you receive, think carefully and reason step-by-step. "
    "Label each reasoning step as 'Step 1:', 'Step 2:', etc. "
    "Keep your reasoning concise, and be brief in each step."
)




ASSISTANT_START = """{thinking_token_open}Let's think step-by-step.\nStep 1: """
# ASSISTANT_START = """<think>\nLet's think step-by-step.\nStep 1: """
# ASSISTANT_START = ""



SYSTEM_MCQ = (
    "When you output the final answer, output ONLY the single letter corresponding to the correct answer. Do not include periods, explanations, or any other text."
)

SYSTEM_COT_ONLY = (
    "During your reasoning, do NOT reveal, hint at, or restate the final answer. "
    "Do not write lines like 'Answer:', 'Final answer:', any answer strings, or any concluding sentence. "
    "Stop immediately after your last numbered reasoning step."
)

SYSTEM_BFCL = (
    "When you output the final answer, output ONLY the function call(s) in the format "
    "[func_name(param=value, ...)] with no other text."
)

SYSTEM_MATH = (
    "When you output the final answer, output ONLY the exact numerical or symbolic answer. "
    "Do not include units or explanations."
)

SYSTEM_BINARY = (
    "When you output the final answer, output ONLY 'Yes' or 'No'. "
    "Do not include periods, explanations, or any other text."
)

SYSTEM_OTHER = (
    "When you output the final answer, output ONLY the answer as a short string. "
    "Do not include explanations or any other text."
)



# ---------------------------------------------------------------------------
# Per-dataset user-turn prompt templates
# The placeholder {question} will be filled at runtime.
# Some datasets expose extra fields (e.g. {context}, {functions}) —
# those are documented in the docstring of each constant.
# ---------------------------------------------------------------------------

# --- BFCL v1 ---------------------------------------------------------------
# Extra fields: {functions}  — JSON description of available API functions.
BFCL_prev = """\
You are given a user request and a list of available API functions.
Your task is to decide which function(s) to call and with what arguments.

Available functions:
{functions}

User request:
{question}

Reason step-by-step through the request, then specify the exact function \
call(s) needed.

Step 1: Identify what the user is asking for.
Step 2: Match the request to the most relevant function(s).
Step 3: Determine the correct argument values from the request.
Step 4: Verify the chosen call satisfies all constraints in the function \
schema.

\\boxed{function call(s) in the required format}
"""

BFCL = """\
You are also an expert in composing functions.You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your FINAL response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the FINAL response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
{functions}

Question:
{question}
"""

# --- BigBench Movie Recommendation -----------------------------------------
BIGBENCH_MOVIE = """\
{question}

Think carefully about the movies listed and the options provided. Reason \
step-by-step before choosing the best answer.
"""



# --- BigBench Causal Judgement ----------------------------------------------
BIGBENCH_CAUSAL = """\
{question}

Reason through the causal relationships step-by-step before answering.
"""

# --- LogiQA ----------------------------------------------------------------
LOGIQIA = """\
You are an expert in logical reasoning.

Reason step-by-step through the logical relationships before selecting an answer.
"""

LOGIQIA_experimental_system = """\
You are an expert in logical reasoning.

Context:
{context}

Question:
{question}

Options:
{options}

Reason step-by-step through the logical relationships before selecting an answer.
"""

# --- CodeQA ----------------------------------------------------------------
# Extra fields: {code}  — the code snippet being asked about.
CODEQA = """\
Code snippet:
```
{code}
```

Question:
{question}

Analyze the code step-by-step and answer the question accurately.
"""

# --- CS1QA -----------------------------------------------------------------
CS1QA = """\
You are an expert computer science instructor helping a student with an \
introductory programming question.

Question:
{question}

Reason step-by-step before answering.

Step 1: Identify the core concept or programming construct being tested.
Step 2: Recall the relevant rules or definitions.
Step 3: Apply those rules to the specific scenario in the question.
Step 4: Arrive at the correct answer and explain it clearly.

"""

# --- CS1QA Type Classification ---------------------------------------------
# Extra fields: {code}  — the code snippet the question is about.
CS1QA_TYPE = """\
You are an expert at classifying introductory programming questions.

Code:
```
{code}
```

Question:
{question}

Classify this question into exactly one of the following types:
code_understanding, logical, error, usage, algorithm, task, comparison, \
reasoning, code_explain, variable, guiding

Reason step-by-step about what the question is asking, then output the \
type label in \\boxed{type_label}.
"""

# --- CS1QA Span Prediction -------------------------------------------------
# Extra fields: {code}  — the code snippet with line numbers prepended.
CS1QA_SPAN = """\
You are an expert at identifying relevant code for programming questions.

Code (with line numbers):
```
{code}
```

Question:
{question}

Identify which lines of code are most relevant to answering this question. \
Reason step-by-step, then output the line range as \\boxed{startLine-endLine} \
(e.g. \\boxed{5-12}).
"""

# --- CS1QA Answer Generation -----------------------------------------------
# Extra fields: {code}  — the code snippet the question is about.
CS1QA_ANSWER = """\
You are an expert computer science instructor helping a student with an \
introductory programming question.

Code:
```
{code}
```

Question:
{question}

Analyze the code step-by-step and answer the question clearly and concisely.

\\boxed{{your answer}}
"""

# --- HotPotQA --------------------------------------------------------------
HOTPOTQA = """\
Question:
{question}

Answer the question by reasoning step-by-step.
"""

# --- College Math Test -----------------------------------------------------
COLLEGE_MATH = """\
Question: {question}

Solve the problem step-by-step, showing all work clearly.
"""

# --- OlympiadBench ---------------------------------------------------------
OLYMPIADBENCH = """\
Question: {question}

Approach the problem rigorously and creatively, step-by-step.
"""

# --- Math500 ---------------------------------------------------------------
MATH500 = """\
Question: {question}

Solve step-by-step, showing all reasoning clearly.
"""

# --- HLE (Humanity's Last Exam) --------------------------------------------
HLE = """\
Question:
{question}

This is a highly challenging question that may require deep, multi-domain \
reasoning. Think carefully and systematically.
"""

# ---------------------------------------------------------------------------
# Registry mapping dataset name → (system_prompt, user_prompt_template)
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: dict[str, tuple[str, str, str]] = {
    "bfcl":                 (SYSTEM, BFCL, ASSISTANT_START),
    "bigbench_movie":       (SYSTEM, BIGBENCH_MOVIE, ASSISTANT_START),
    "bigbench_causal":      (SYSTEM, BIGBENCH_CAUSAL, ASSISTANT_START),
    "logiqа":               (SYSTEM, LOGIQIA, ASSISTANT_START),   # kept as alias below too
    "logiqa":               (SYSTEM, LOGIQIA, ASSISTANT_START),
    "codeqa":               (SYSTEM, CODEQA, ASSISTANT_START),
    "cs1qa":                (SYSTEM, CS1QA, ASSISTANT_START),
    "cs1qa_type":           (SYSTEM, CS1QA_TYPE, ASSISTANT_START),
    "cs1qa_span":           (SYSTEM, CS1QA_SPAN, ASSISTANT_START),
    "cs1qa_answer":         (SYSTEM, CS1QA_ANSWER, ASSISTANT_START),
    "hotpotqa":             (SYSTEM, HOTPOTQA, ASSISTANT_START),
    "college_math_test":    (SYSTEM, COLLEGE_MATH, ASSISTANT_START),
    "olympiadbench":        (SYSTEM, OLYMPIADBENCH, ASSISTANT_START),
    "math500":              (SYSTEM, MATH500, ASSISTANT_START),
    "hle":                  (SYSTEM, HLE, ASSISTANT_START),
}

DATASET_ANSWER_FORMAT: dict[str, str] = {
    "bfcl":              SYSTEM_BFCL,
    "bigbench_movie":    SYSTEM_MCQ,
    "bigbench_causal":   SYSTEM_BINARY,
    "logiqa":            SYSTEM_MCQ,
    "college_math":      SYSTEM_MATH,
    "college_math_test": SYSTEM_MATH,
    "math500":           SYSTEM_MATH,
    "olympiadbench":     SYSTEM_MATH,
    "hotpotqa":          SYSTEM_OTHER,
}
