"""
Few-shot Chain-of-Thought prompts for each supported dataset.
"""


# --- LogiQA ----------------------------------------------------------------
# Sourced from eval.txt at https://github.com/lgw863/LogiQA-dataset (the official github repo for LogiQA)
LOGIQA_FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Context:\n"
            "Black Americans are twice as likely to suffer from hypertension as white Americans."
            "The same is true when comparing Westernized black Africans to white Africans."
            "The researchers hypothesized that the reason why westernized black people suffer from hypertension"
            " is the result of the interaction of two reasons? one is the high salt content of western foods,"
            " and the other is the adaptation mechanism of black genetic genes to the salt-deficient environment.\n"
            "\n"
            "Question:\n"
            "The following conclusions about contemporary westernized African blacks, if the item is true,"
            " can it best support the researchers' hypothesis?\n"
            "\n"
            "Options:\n"
            "A. The blood pressure of the descendants of Senegalese and Gambians is usually not high,"
            " and the history of Senegal and Gambia has not been short of salt.\n"
            "B. The unusually high salt intake in certain parts of Africa is a serious problem"
            " that threatens the health of residents.\n"
            "C. Considering health care, most African whites also pay attention to controlling salt intake.\n"
            "D. The blood pressure of Yoruba people in West Africa is not high."
            " Yoruba people have lived inland far away from sea salt and far away from the Sahara salt mine in Africa."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Let's think step-by-step.\n"
            "Step 1: From the hypothesis, Hypertension = (Genetic trait to save salt) + (Western high-salt diet).\n"
            "Step 2: Identify the Necessary Evidence: I need to find a group where the \"Genetic trait\""
            " is missing to see if the \"Hypertension\" also disappears.\n"
            "Step 3: From Option A, Residents of Senegal/Gambia had plenty of salt (no need for the gene)."
            " Their descendants in the West have normal blood pressure.\n"
            "Step 4: This confirms that when the \"salt-deficient\" genetic history is removed,"
            " the Western diet doesn't cause the same level of hypertension.\n"
            "Step 5: Option A is the best fit.\n"
            "Final Answer: A"
        ),
    },
    {
        "role": "user",
        "content": (
            "Context:\n"
            "The prohibition of advertising cigarettes on public media does not reduce the number of young people"
            " smoking, because they have known for a long time that there are cigarettes in the world,"
            " known for various brands of cigarettes, and know where to get them."
            " They do not need advertisements to provide this information.\n"
            "\n"
            "Question:\n"
            "The following, if true, can weaken the above argument most?\n"
            "\n"
            "Options:\n"
            "A. Watching or listening to advertisements can increase a person's desire to obtain such products.\n"
            "B. Prohibition of cigarette advertisements on public media will cause other forms"
            " of cigarette advertisements to proliferate.\n"
            "C. Cigarette advertising on public media is a major expense for tobacco companies.\n"
            "D. Anti-smokers have advertised in the public media against smoking from the beginning."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Let's think step-by-step.\n"
            "Step 1: The argument claims that ads don't work because young people already know cigarettes exist"
            " and where to buy them. It assumes ads only function as a source of \"missing information.\"\n"
            "Step 2: The argument overlooks that advertising isn't just about providing information;"
            " it's about persuasion and desire.\n"
            "Step 3: To weaken this, we need an option that shows advertisements do more than just tell people"
            " \"cigarettes are for sale\"—they actually drive the urge to smoke.\n"
            "Step 4: A directly addresses this by stating ads increase the desire to obtain the product."
            " B and C discuss the consequences for companies, not the behavior of young people."
            " D talks about anti-smoking ads, which doesn't explain why cigarette ads themselves are effective or ineffective.\n"
            "Step 5: Option A is the best choice to weaken the argument.\n"
            "Final Answer: A"
        ),
    },
]







# --- CodeQA ----------------------------------------------------------------
CODEQA_FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Code snippet:\n"
            "```python\n"
            "def find_duplicates(nums):\n"
            "    seen = set()\n"
            "    duplicates = []\n"
            "    for n in nums:\n"
            "        if n in seen:\n"
            "            duplicates.append(n)\n"
            "        else:\n"
            "            seen.add(n)\n"
            "    return duplicates\n"
            "```\n"
            "\n"
            "Question:\n"
            "What does `find_duplicates([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])` return?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Step 1: Understand the function's purpose. It iterates over `nums`, tracking elements in `seen`."
            " If an element is already in `seen`, it is appended to `duplicates`; otherwise it is added to `seen`.\n"
            "Step 2: Trace through the input `[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]` step by step:\n"
            "  - n=3: not in seen → seen={3}\n"
            "  - n=1: not in seen → seen={3,1}\n"
            "  - n=4: not in seen → seen={3,1,4}\n"
            "  - n=1: 1 in seen → duplicates=[1]\n"
            "  - n=5: not in seen → seen={3,1,4,5}\n"
            "  - n=9: not in seen → seen={3,1,4,5,9}\n"
            "  - n=2: not in seen → seen={3,1,4,5,9,2}\n"
            "  - n=6: not in seen → seen={3,1,4,5,9,2,6}\n"
            "  - n=5: 5 in seen → duplicates=[1,5]\n"
            "  - n=3: 3 in seen → duplicates=[1,5,3]\n"
            "Step 3: The function returns `duplicates` after the loop ends.\n"
            "Step 4: Note that a value is appended each time it is seen again, so if an element appeared"
            " three times it would appear twice in the output. In this input every duplicate appears exactly twice,"
            " so each duplicate is appended exactly once.\n"
            "Final Answer: [1, 5, 3]"
        ),
    },
    {
        "role": "user",
        "content": (
            "Code snippet:\n"
            "```python\n"
            "def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
            "```\n"
            "\n"
            "Question:\n"
            "What is the time complexity of `flatten` in terms of N, the total number of elements"
            " (both lists and non-list values) across all levels of nesting?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Step 1: Identify what work is done per recursive call."
            " Each call iterates over the direct children of the list it receives."
            " For each child that is itself a list, it makes a recursive call and then calls `extend`"
            " to append the returned elements into `result`.\n"
            "Step 2: Consider the cost of `extend`. Extending by k elements takes O(k) time."
            " In the worst case (a deeply nested structure), the results of inner calls are extended"
            " into outer results multiple times — once per level of nesting above them.\n"
            "Step 3: Analyze the worst case. For a linear chain of depth D with one leaf at the bottom,"
            " the single leaf value is 'extended' at every level, giving O(D) total work just for that leaf."
            " With N total nodes and depth up to N, this yields O(N^2) in the worst case"
            " (e.g., `[[[...[[x]]...]]]` with N-1 single-element wrapper lists).\n"
            "Step 4: Verify the best case. For a completely flat list (depth 1),"
            " every item is a non-list and is simply appended in O(1). Total work is O(N).\n"
            "Step 5: The overall time complexity is therefore O(N^2) in the worst case"
            " and O(N) in the best case (flat input).\n"
            "Final Answer: O(N^2) worst case (due to repeated extend calls propagating elements up each level"
            " of nesting), O(N) best case for a flat list."
        ),
    },
]


FEW_SHOT_PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    # "bfcl":                 (SYSTEM, BFCL, ASSISTANT_START),
    # "bigbench_movie":       (SYSTEM, BIGBENCH_MOVIE, ASSISTANT_START),
    # "bigbench_causal":      (SYSTEM, BIGBENCH_CAUSAL, ASSISTANT_START),
    "logiqa": LOGIQA_FEW_SHOT_MESSAGES,
    # "logiqa":               (SYSTEM, LOGIQIA, ASSISTANT_START),
    "codeqa": CODEQA_FEW_SHOT_MESSAGES,
    # "cs1qa":                (SYSTEM, CS1QA, ASSISTANT_START),
    # "hotpotqa":             (SYSTEM, HOTPOTQA, ASSISTANT_START),
    # "college_math_test":    (SYSTEM, COLLEGE_MATH, ASSISTANT_START),
    # "olympiadbench":        (SYSTEM, OLYMPIADBENCH, ASSISTANT_START),
    # "math500":              (SYSTEM, MATH500, ASSISTANT_START),
    # "hle":                  (SYSTEM, HLE, ASSISTANT_START),
}


