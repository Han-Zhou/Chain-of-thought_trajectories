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
    "For every problem you receive, think carefully and reason step-by-step "
    "before giving your final answer. "
    "Label each reasoning step as 'Step 1:', 'Step 2:', etc. "
    "After all steps, write your final answer on a new line starting with "
    "'Final Answer:'."
)


ASSISTANT_START = """Let's think step-by-step.\nStep 1: """
# ASSISTANT_START = """<think>\nLet's think step-by-step.\nStep 1: """
# ASSISTANT_START = ""


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

Final Answer: <function call(s) in the required format>
"""

BFCL = """\
You are also an expert in composing functions.You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your FINAL response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  You SHOULD NOT include any other text in the FINAL response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
{functions}
"""

# --- BigBench Movie Recommendation -----------------------------------------
BIGBENCH_MOVIE = """\
You are a movie recommendation expert.

Question:
{question}

Think carefully about the question and the options provided. Reason \
step-by-step before choosing the best answer.

Step 1: Understand what the question is asking.
Step 2: Analyze each candidate answer against the question criteria.
Step 3: Eliminate implausible options with justification.
Step 4: Select the best answer.

Final Answer: <your answer>
"""

# --- BigBench Causal Judgement ----------------------------------------------
BIGBENCH_CAUSAL = """\
You are an expert in causal reasoning.

Question:
{question}

Reason through the causal relationships step-by-step before answering.

Step 1: Identify the events or entities involved.
Step 2: Determine the cause-and-effect relationships described.
Step 3: Apply causal reasoning principles (counterfactual, mechanism, etc.).
Step 4: Reach a conclusion based on your causal analysis.

Final Answer: <Yes / No / your conclusion>
"""

# --- LogiQA ----------------------------------------------------------------
LOGIQIA = """\
You are an expert in logical reasoning.

Reason step-by-step through the logical relationships before selecting an answer. 'Final Answer:' must be followed by exactly one letter: A, B, C, or D.
"""

LOGIQIA_experimental_system = """\
You are an expert in logical reasoning.

Context:
{context}

Question:
{question}

Options:
{options}

Reason step-by-step through the logical relationships before selecting an answer. 'Final Answer:' must be followed by exactly one letter: A, B, C, or D.
"""

# --- CodeQA ----------------------------------------------------------------
# Extra fields: {code}  — the code snippet being asked about.
CODEQA = """\
You are an expert software engineer and code analyst.

Code snippet:
```
{code}
```

Question:
{question}

Analyze the code step-by-step and answer the question accurately.

Step 1: Understand the overall purpose of the code.
Step 2: Trace through the relevant logic or data flow.
Step 3: Identify any edge cases or important behavior related to the question.
Step 4: Formulate a precise answer based on your analysis.

Final Answer: <your answer>
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

Final Answer: <your answer>
"""

# --- HotPotQA --------------------------------------------------------------
# Extra fields: {context}  — the supporting passages.
HOTPOTQA = """\
You are an expert at multi-hop question answering.

Supporting passages:
{context}

Question:
{question}

Answer the question by reasoning across the passages step-by-step.

Step 1: Identify the key entities and relationships in the question.
Step 2: Locate relevant information in the supporting passages.
Step 3: Connect information across passages to bridge reasoning hops.
Step 4: Synthesize the evidence to form a precise answer.

Final Answer: <your answer>
"""

# --- College Math Test -----------------------------------------------------
COLLEGE_MATH = """\
You are an expert mathematician.

Problem:
{question}

Solve the problem step-by-step, showing all work clearly.

Step 1: Identify the mathematical domain and key concepts involved.
Step 2: Write down any relevant formulas, theorems, or definitions.
Step 3: Set up the solution approach.
Step 4: Execute the computation or proof, showing each sub-step.
Step 5: Verify the result (check edge cases, units, plausibility).

Final Answer: <exact answer>
"""

# --- OlympiadBench ---------------------------------------------------------
OLYMPIADBENCH = """\
You are an expert at mathematical olympiad problems.

Problem:
{question}

Approach the problem rigorously and creatively, step-by-step.

Step 1: Carefully read and restate the problem in your own words.
Step 2: Identify the key constraints and what needs to be proven or found.
Step 3: Explore relevant strategies (e.g., invariants, constructions, \
induction, algebra, combinatorics).
Step 4: Develop and execute the chosen strategy in detail.
Step 5: Verify the solution satisfies all given conditions.

Final Answer: <exact answer or proof conclusion>
"""

# --- Math500 ---------------------------------------------------------------
MATH500 = """\
You are an expert mathematician.

Problem:
{question}

Solve step-by-step, showing all reasoning clearly.

Step 1: Identify the problem type and relevant mathematical concepts.
Step 2: Recall applicable formulas or theorems.
Step 3: Set up the solution, defining variables and relationships.
Step 4: Carry out the computation or derivation step-by-step.
Step 5: State and verify the final answer.

Final Answer: <exact answer in simplified form>
"""

# --- HLE (Humanity's Last Exam) --------------------------------------------
HLE = """\
You are a world-class expert across all academic disciplines.

Question:
{question}

This is a highly challenging question that may require deep, multi-domain \
reasoning. Think carefully and systematically.

Step 1: Determine the domain(s) relevant to this question.
Step 2: Recall and state the key facts, principles, or theories involved.
Step 3: Reason through the question carefully, considering all angles.
Step 4: Evaluate potential answers critically.
Step 5: Select or construct the most accurate and well-supported answer.

Final Answer: <your answer>
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
    "hotpotqa":             (SYSTEM, HOTPOTQA, ASSISTANT_START),
    "college_math_test":    (SYSTEM, COLLEGE_MATH, ASSISTANT_START),
    "olympiadbench":        (SYSTEM, OLYMPIADBENCH, ASSISTANT_START),
    "math500":              (SYSTEM, MATH500, ASSISTANT_START),
    "hle":                  (SYSTEM, HLE, ASSISTANT_START),
}
