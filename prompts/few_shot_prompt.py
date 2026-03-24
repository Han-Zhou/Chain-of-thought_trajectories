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
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: From the hypothesis, Hypertension = (Genetic trait to save salt) + (Western high-salt diet).\n"
            "Step 2: Identify the Necessary Evidence: I need to find a group where the \"Genetic trait\""
            " is missing to see if the \"Hypertension\" also disappears.\n"
            "Step 3: From Option A, Residents of Senegal/Gambia had plenty of salt (no need for the gene)."
            " Their descendants in the West have normal blood pressure.\n"
            "Step 4: This confirms that when the \"salt-deficient\" genetic history is removed,"
            " the Western diet doesn't cause the same level of hypertension.\n"
            "Step 5: Option A is the best fit.{thinking_token_close}\n"
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
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The argument claims that ads don't work because young people already know cigarettes exist"
            " and where to buy them. It assumes ads only function as a source of \"missing information.\"\n"
            "Step 2: The argument overlooks that advertising isn't just about providing information;"
            " it's about persuasion and desire.\n"
            "Step 3: To weaken this, we need an option that shows advertisements do more than just tell people"
            " \"cigarettes are for sale\"—they actually drive the urge to smoke.\n"
            "Step 4: A directly addresses this by stating ads increase the desire to obtain the product."
            " B and C discuss the consequences for companies, not the behavior of young people."
            " D talks about anti-smoking ads, which doesn't explain why cigarette ads themselves are effective or ineffective.\n"
            "Step 5: Option A is the best choice to weaken the argument.{thinking_token_close}\n"
            "Final Answer: A"
        ),
    },
]


# BigBench Movie Recommendation
# Sourced from the training split from tasksource/bigbench
BIGBENCH_MOVIE = [
    {
        "role": "user",
        "content": (
            "Find a movie similar to Batman, The Mask, The Fugitive, Pretty Woman:\n"
            "Options:\n"
            "A. Maelstrom\n"
            "B. The Lion King\n"
            "C. Lamerica\n"
            "D. The Front Page"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: Identify the pattern. Batman (1989), The Mask (1994), The Fugitive (1993),"
            " and Pretty Woman (1990) are all major Hollywood blockbusters from the late 80s/early 90s"
            " that were huge box office hits with broad mainstream appeal.\n"
            "Step 2: Evaluate each option against this pattern.\n"
            "Step 3: Maelstrom is an obscure Canadian art film — does not fit.\n"
            "Step 4: Lamerica is an Italian art house film — does not fit.\n"
            "Step 5: The Front Page is a 1974 comedy — wrong era and profile.\n"
            "Step 6: The Lion King (1994) is a massive mainstream Hollywood blockbuster from the same era,"
            " fitting perfectly with the pattern of widely popular, commercially successful early-90s films.\n"
            "Step 7: The Lion King is the best match.{thinking_token_close}\n"
            "Final Answer: B"
        ),
    },
    {
        "role": "user",
        "content": (
            "Find a movie similar to The Sixth Sense, The Matrix, Forrest Gump, The Shawshank Redemption:\n"
            "Options:\n"
            "A. Terminator 2 Judgment Day\n"
            "B. The Sheltering Sky\n"
            "C. Street Fighter II The Animated Movie\n"
            "D. The Boy Who Could Fly"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: These are all iconic, critically acclaimed Hollywood films from the 1990s"
            " that were both massive commercial successes and culturally influential.\n"
            "Step 2: Evaluate each option against this pattern.\n"
            "Step 3: The Sheltering Sky is a Bernardo Bertolucci art film — does not fit the mainstream blockbuster pattern.\n"
            "Step 4: Street Fighter II The Animated Movie is a niche anime adaptation — does not fit.\n"
            "Step 5: The Boy Who Could Fly is a small 1986 fantasy film — wrong era and scale.\n"
            "Step 6: Terminator 2: Judgment Day (1991) is a hugely popular, critically acclaimed,"
            " culturally iconic blockbuster from the same era — a perfect match.\n"
            "Step 7: Terminator 2 Judgment Day is the best match.{thinking_token_close}\n"
            "Final Answer: A"
        ),
    },
]



# BigBench Causal Judgement
# Sourced from the training split from tasksource/bigbench
BIGBENCH_CAUSAL = [
    {
        "role": "user",
        "content": (
            "How would a typical person answer each of the following questions about causation?\n"
            "\n"
            "Q: Lauren and Jane work for the same company. They each need to use a computer for work sometimes. "
            "Unfortunately, the computer isn't very powerful. If two people are logged on at the same time, "
            "it usually crashes. So the company decided to institute an official policy. It declared that Lauren "
            "would be the only one permitted to use the computer in the mornings and that Jane would be the only "
            "one permitted to use the computer in the afternoons. As expected, Lauren logged on the computer the "
            "next day at 9:00 am. But Jane decided to disobey the official policy. She also logged on at 9:00 am. "
            "The computer crashed immediately. Did Jane cause the computer to crash?\n"
            "Options:\n"
            "- Yes\n"
            "- No"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The question asks how a typical person would judge causation here.\n"
            "Step 2: The company policy says Lauren is allowed on in the morning and Jane is not. "
            "Lauren followed the rules; Jane violated the policy by logging on at 9:00 am.\n"
            "Step 3: The computer crashes when two people are logged on simultaneously. "
            "Both Lauren and Jane being logged on caused the crash in a physical sense.\n"
            "Step 4: However, a typical person assigns causal blame based on norm violation. "
            "Lauren was acting within the rules, while Jane was breaking the official policy.\n"
            "Step 5: Because Jane's action was the norm-violating one, a typical person would say "
            "Jane caused the crash.{thinking_token_close}\n"
            "Final Answer: Yes"
        ),
    },
    {
        "role": "user",
        "content": (
            "How would a typical person answer each of the following questions about causation?\n"
            "\n"
            "Q: David has a new dryer in his apartment. David's clothes will dry in sixty minutes if either "
            "the cycle is set to MAX DRY or the temperature is set to HIGH. Today, the cycle is already set "
            "to MAX DRY, and the temperature is not set on HIGH. David checks the dryer's settings, and he "
            "sees that the temperature is not set on HIGH. He completely changes the setting, and he sets the "
            "temperature to HIGH. He then turns on the dryer. Because the dryer would dry David's clothes in "
            "sixty minutes if either the cycle is set to MAX DRY or the temperature is set to HIGH, the dryer "
            "dries David's clothes in sixty minutes. Did David's clothes dry in sixty minutes because David "
            "changed the temperature setting?\n"
            "Options:\n"
            "- Yes\n"
            "- No"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The question asks whether David changing the temperature caused the clothes to dry in sixty minutes.\n"
            "Step 2: The dryer dries in sixty minutes if EITHER MAX DRY is on OR HIGH temperature is on. "
            "This is a disjunctive (OR) causal structure.\n"
            "Step 3: The cycle was already set to MAX DRY before David did anything. "
            "That alone was sufficient to dry the clothes in sixty minutes.\n"
            "Step 4: David then also set the temperature to HIGH, adding a second sufficient condition. "
            "But removing his action wouldn't change the outcome — the clothes would still dry in sixty minutes "
            "because MAX DRY was already on.\n"
            "Step 5: A typical person recognizes that David's change was redundant. The clothes would have dried "
            "in sixty minutes regardless. So his action is not seen as the cause.{thinking_token_close}\n"
            "Final Answer: No"
        ),
    },
]


# CodeQA
# Sourced from the first and second datapoint from vm2825/CodeQA-dataset
CODEQA_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Code snippet:\n"
            "```\n"
            "def MakeSuiteFromDict(d, label=None):\n"
            "\tsuite = Suite(label=label)\n"
            "\tsuite.SetDict(d)\n"
            "\tsuite.Normalize()\n"
            "\treturn suite\n"
            "```\n"
            "\n"
            "Question:\n"
            "What does the code make?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The function is called MakeSuiteFromDict, which suggests it creates some kind of suite object.\n"
            "Step 2: Inside the function, it instantiates a Suite object with an optional label parameter.\n"
            "Step 3: It then populates the suite by calling SetDict(d) with the input dictionary "
            "and normalizes it with Normalize().\n"
            "Step 4: Finally, it returns the suite object. "
            "The code makes a suite.{thinking_token_close}\n"
            "Final Answer: a suite"
        ),
    },
    {
        "role": "user",
        "content": (
            "Code snippet:\n"
            "```\n"
            "def MakeSuiteFromDict(d, label=None):\n"
            "\tsuite = Suite(label=label)\n"
            "\tsuite.SetDict(d)\n"
            "\tsuite.Normalize()\n"
            "\treturn suite\n"
            "```\n"
            "\n"
            "Question:\n"
            "Does the code make a suite?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The question asks whether the code makes a suite, so I need to check "
            "if a Suite object is created.\n"
            "Step 2: On the first line of the function body, suite = Suite(label=label) "
            "creates a new Suite instance.\n"
            "Step 3: The function then configures it with SetDict and Normalize, "
            "and returns it.\n"
            "Step 4: Yes, the code does make a suite.{thinking_token_close}\n"
            "Final Answer: Yes"
        ),
    },
]








# CS1QA
# Sourced from entries 209 and 87 of the train split from cs1qa/train_cleaned.jsonl
CS1QA_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Code snippet:\n"
            "```\n"
            "def fibonacci(upper_bound):\n"
            "    sequence=[0,1]\n"
            "    n=0\n"
            "    while (sequence[n]+sequence[n+1])<1000:\n"
            "        sequence.append(sequence[n]+sequence[n+1])\n"
            "        n+=1\n"
            "    return sequence\n"
            "print(fibonacci(1000))\n"
            "```\n"
            "\n"
            "Question:\n"
            "Could you please explain what append does in Task 1?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The question asks about the append method used inside the "
            "fibonacci function.\n"
            "Step 2: In Python, list.append(x) adds the element x to the end of "
            "the list. It modifies the list in place and does not return a new list.\n"
            "Step 3: In this code, sequence.append(sequence[n]+sequence[n+1]) "
            "computes the sum of the two most recent Fibonacci numbers and adds "
            "the result as the next element of the sequence list.\n"
            "Step 4: So append here serves to grow the Fibonacci sequence one "
            "element at a time, adding each new Fibonacci number to the end of "
            "the list to be returned.{thinking_token_close}\n"
            "Final Answer: The append function adds an element to the end of "
            "the list. It serves to add a new Fibonacci number to the list to "
            "be returned."
        ),
    },
    {
        "role": "user",
        "content": (
            "Code snippet:\n"
            "```\n"
            "from cs1robots import *\n"
            "create_world()\n"
            "hubo=Robot(orientation='W',avenue=7, street=5)\n"
            "hubo.set_trace(\"blue\")\n"
            "\n"
            "def turn_right():\n"
            "    hubo.turn_left()\n"
            "    hubo.turn_left()\n"
            "    hubo.turn_left()\n"
            "\n"
            "while not hubo.facing_north():\n"
            " hubo.turn_left()\n"
            "```\n"
            "\n"
            "Question:\n"
            "Is hubo.facing_north() also true or false?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The question asks whether hubo.facing_north() returns a "
            "boolean value (True or False).\n"
            "Step 2: In the cs1robots library, methods like facing_north() are "
            "predicate methods — they test a condition about the robot's state "
            "and return True or False.\n"
            "Step 3: In the code, it is used as 'while not hubo.facing_north()', "
            "which confirms it returns a boolean that can be negated with 'not'.\n"
            "Step 4: The direction checked is an absolute compass direction "
            "(north), so the result depends only on the robot's current heading, "
            "not its initial orientation.{thinking_token_close}\n"
            "Final Answer: Yes, it returns True or False. It is True when "
            "the robot is facing north, and the direction is absolute, so "
            "it is not affected by Hubo's initial orientation."
        ),
    },
]


# --- HotPotQA ---------------------------------------------------------------
# Sourced from the first two datapoints of hotpotqa/hotpot_qa validation (fullwiki)
HOTPOTQA_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Context:\n"
            "Arthur's Magazine: Arthur's Magazine (1844–1846) was an American literary periodical "
            "published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work "
            "by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. "
            "In May 1846 it was merged into \"Godey's Lady's Book\".\n\n"
            "First for Women: First for Women is a woman's magazine published by Bauer Media Group "
            "in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. "
            "In 2011 the circulation of the magazine was 1,310,696 copies.\n\n"
            "Question:\n"
            "Which magazine was started first Arthur's Magazine or First for Women?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The question asks which of two magazines was started first. "
            "I need to find the start date of each magazine from the supporting passages.\n"
            "Step 2: From the Arthur's Magazine passage: \"Arthur's Magazine (1844–1846)\" — "
            "it was started in 1844.\n"
            "Step 3: From the First for Women passage: \"The magazine was started in 1989.\"\n"
            "Step 4: Comparing the two dates, 1844 is earlier than 1989. "
            "Therefore Arthur's Magazine was started first.{thinking_token_close}\n"
            "Final Answer: Arthur's Magazine"
        ),
    },
    {
        "role": "user",
        "content": (
            "Context:\n"
            "Oberoi family: The Oberoi family is an Indian family that is famous for its involvement "
            "in hotels, namely through The Oberoi Group.\n\n"
            "The Oberoi Group: The Oberoi Group is a hotel company with its head office in Delhi. "
            "Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise "
            "ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.\n\n"
            "Question:\n"
            "The Oberoi family is part of a hotel company that has a head office in what city?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The question asks for the city where the Oberoi family's hotel company "
            "has its head office. This requires connecting information across two passages.\n"
            "Step 2: From the Oberoi family passage: the family is famous for its involvement "
            "in hotels \"through The Oberoi Group.\" This identifies the hotel company.\n"
            "Step 3: From The Oberoi Group passage: \"The Oberoi Group is a hotel company with "
            "its head office in Delhi.\"\n"
            "Step 4: Bridging these two facts — the Oberoi family is part of The Oberoi Group, "
            "and The Oberoi Group's head office is in Delhi.{thinking_token_close}\n"
            "Final Answer: Delhi"
        ),
    },
]


# College Math Test
# Sourced from the first two datapoints of di-zhang-fdu/College_Math_Test
COLLEGE_MATH_FEW_SHOT = [
    {
        "role": "user",
        "content": "Question: Simplify: $-10-4(n-5)$",
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: Distribute $-4$ across $(n-5)$: $-4 \\cdot n = -4n$ and $-4 \\cdot (-5) = 20$.\n"
            "Step 2: Rewrite the expression: $-10 - 4n + 20$.\n"
            "Step 3: Combine the constant terms: $-10 + 20 = 10$.\n"
            "Step 4: The simplified expression is $10 - 4n$.{thinking_token_close}\n"
            "Final Answer: $10-4 n$"
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: Solve the following systems of equation:\n"
            "$x y=90$\n"
            "$(x-5)(y+1)=120$"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: From the first equation, $xy = 90$, so $y = \\frac{90}{x}$.\n"
            "Step 2: Expand the second equation: $(x-5)(y+1) = xy + x - 5y - 5 = 120$.\n"
            "Step 3: Substitute $xy = 90$: $90 + x - 5y - 5 = 120$, which simplifies to $x - 5y = 35$.\n"
            "Step 4: Substitute $y = \\frac{90}{x}$: $x - \\frac{450}{x} = 35$.\n"
            "Step 5: Multiply through by $x$: $x^2 - 35x - 450 = 0$.\n"
            "Step 6: Factor: $(x - 45)(x + 10) = 0$, so $x = 45$ or $x = -10$.\n"
            "Step 7: When $x = 45$, $y = \\frac{90}{45} = 2$. "
            "When $x = -10$, $y = \\frac{90}{-10} = -9$.\n"
            "Step 8: The solutions are $(45, 2)$ and $(-10, -9)$.{thinking_token_close}\n"
            "Final Answer: $(45,2),(-10,-9)$"
        ),
    },
]


# OlympiadBench
# Sourced from the first two datapoints of Hothan/OlympiadBench OE_TO_maths_en_COMP
OLYMPIADBENCH_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Question: Xenia and Sergey play the following game. Xenia thinks of a positive integer $N$ "
            "not exceeding 5000. Then she fixes 20 distinct positive integers $a_{1}, a_{2}, "
            "\\ldots, a_{20}$ such that, for each $k=1,2, \\ldots, 20$, the numbers $N$ and "
            "$a_{k}$ are congruent modulo $k$. By a move, Sergey tells Xenia a set $S$ of "
            "positive integers not exceeding 20, and she tells him back the set "
            "$\\{a_{k}: k \\in S\\}$ without spelling out which number corresponds to which "
            "index. How many moves does Sergey need to determine for sure the number "
            "Xenia thought of?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: Sergey needs to recover $N \\pmod{m}$ for a modulus $m > 5000$ "
            "so that $N$ is uniquely determined. By the Chinese Remainder Theorem, "
            "he should aim to learn $a_k$ for indices whose product of moduli exceeds 5000.\n"
            "Step 2 (2 moves suffice): On the first move Sergey queries $S_1 = \\{17, 18\\}$, "
            "and on the second move $S_2 = \\{18, 19\\}$. The value $a_{18}$ appears in both "
            "responses, so Sergey identifies $a_{17}$, $a_{18}$, and $a_{19}$. "
            "Since $17 \\cdot 18 \\cdot 19 = 5814 > 5000$, CRT uniquely determines $N$.\n"
            "Step 3 (1 move does not suffice): Let $M = \\operatorname{lcm}(1,2,\\ldots,10) = 2520$. "
            "For any query set $S = \\{s_1,\\ldots,s_k\\}$, one can construct distinct values "
            "$b_1,\\ldots,b_k$ with $b_i \\equiv 1 \\pmod{s_i}$ and "
            "$b_i \\equiv M+1 \\pmod{s_{i+1}}$ (indices mod $k$). "
            "This is possible because $M$ is divisible by $\\gcd(s_i, s_{i+1})$ for every pair. "
            "Xenia can return $\\{b_1,\\ldots,b_k\\}$ making $N=1$ and $N=M+1=2521$ "
            "indistinguishable, so one move is insufficient.\n"
            "Step 4: Therefore the answer is 2.{thinking_token_close}\n"
            "Final Answer: 2"
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: Given a positive integer $n$, determine the largest real number $\\mu$ "
            "satisfying the following condition: for every $4n$-point configuration $C$ "
            "in an open unit square $U$, there exists an open rectangle in $U$, whose sides "
            "are parallel to those of $U$, which contains exactly one point of $C$, and has "
            "an area greater than or equal to $\\mu$."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1 (Upper bound): Place $n$ clusters of 4 points in $U = (0,1)^2$ at "
            "$(\\frac{i}{n+1} \\pm \\epsilon) \\times (\\frac{1}{2} \\pm \\epsilon)$ for "
            "$i = 1,\\ldots,n$. Any axis-aligned rectangle in $U$ containing exactly one "
            "point has area at most $(\\frac{1}{n+1}+\\epsilon)(\\frac{1}{2}+\\epsilon)$, "
            "which approaches $\\frac{1}{2(n+1)}$ as $\\epsilon \\to 0$. "
            "So $\\mu \\leq \\frac{1}{2(n+1)}$.\n"
            "Step 2 (Lower bound): Set $\\mu_0 = \\frac{2}{|C|+4} = \\frac{2}{4n+4} = \\frac{1}{2(n+1)}$. "
            "Project $C$ onto the $x$-axis to get distinct values $x_1 < \\cdots < x_k$, "
            "with $m_i$ points on each vertical line $x = x_i$.\n"
            "Step 3: Using a lemma on isolating points in an interval — if "
            "$x_{i+1} - x_{i-1} > (\\lfloor m_i/2 \\rfloor + 1)\\mu_0$, one can isolate "
            "a point on line $x_i$ by a vertical sub-interval of length "
            "$\\geq \\mu_0 / (x_{i+1}-x_{i-1})$, giving a rectangle of area $\\geq \\mu_0$.\n"
            "Step 4: A telescoping-sum argument using the inequality "
            "$\\lfloor m_1/2 \\rfloor + \\sum \\lfloor m_i/2 \\rfloor + \\lfloor m_k/2 \\rfloor "
            "\\leq \\sum m_i - k + 2$ shows the opposite case leads to $2 < 2$, a contradiction.\n"
            "Step 5: Therefore $\\mu = \\frac{1}{2(n+1)}$.{thinking_token_close}\n"
            "Final Answer: $\\frac{1}{2 n+2}$"
        ),
    },
]


# MATH-500
# Sourced from the first two datapoints of HuggingFaceH4/MATH-500
MATH500_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Question: Convert the point $(0,3)$ in rectangular coordinates to polar "
            "coordinates. Enter your answer in the form $(r,\\theta),$ where $r > 0$ "
            "and $0 \\le \\theta < 2 \\pi.$"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: Find $r$. We have $r = \\sqrt{x^2 + y^2} = \\sqrt{0^2 + 3^2} = 3$.\n"
            "Step 2: Find $\\theta$. The point $(0, 3)$ lies on the positive $y$-axis, "
            "so the angle with the positive $x$-axis is $\\frac{\\pi}{2}$.\n"
            "Step 3: Therefore the polar coordinates are "
            "$\\left( 3, \\frac{\\pi}{2} \\right)$.{thinking_token_close}\n"
            "Final Answer: $\\left( 3, \\frac{\\pi}{2} \\right)$"
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: Define\n"
            "\\[p = \\sum_{k = 1}^\\infty \\frac{1}{k^2} \\quad \\text{and} \\quad "
            "q = \\sum_{k = 1}^\\infty \\frac{1}{k^3}.\\]\n"
            "Find a way to write\n"
            "\\[\\sum_{j = 1}^\\infty \\sum_{k = 1}^\\infty \\frac{1}{(j + k)^3}\\]\n"
            "in terms of $p$ and $q.$"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: Count how many times $\\frac{1}{n^3}$ appears in the double sum "
            "by conditioning on $j + k = n$. The valid pairs are "
            "$(1, n-1), (2, n-2), \\ldots, (n-1, 1)$, giving $n - 1$ terms.\n"
            "Step 2: Rewrite the double sum: "
            "$\\sum_{j=1}^\\infty \\sum_{k=1}^\\infty \\frac{1}{(j+k)^3} "
            "= \\sum_{n=2}^\\infty \\frac{n-1}{n^3}$.\n"
            "Step 3: Split the fraction: "
            "$\\sum_{n=2}^\\infty \\frac{n-1}{n^3} "
            "= \\sum_{n=2}^\\infty \\frac{1}{n^2} - \\sum_{n=2}^\\infty \\frac{1}{n^3}$.\n"
            "Step 4: Since the $n=1$ terms are $\\frac{1}{1^2} = 1$ and $\\frac{1}{1^3} = 1$, "
            "adding them back cancels: "
            "$\\left(p - 1\\right) - \\left(q - 1\\right) = p - q$.\n"
            "Step 5: The answer is $p - q$.{thinking_token_close}\n"
            "Final Answer: $p - q$"
        ),
    },
]


# HLE (Humanity's Last Exam)
# Sourced from two datapoints of cais/hle
HLE_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Question: Which condition of Arrhenius's sixth impossibility theorem "
            "do critical-level views violate?\n"
            "\n"
            "Answer Choices:\n"
            "A. Egalitarian Dominance\n"
            "B. General Non-Extreme Priority\n"
            "C. Non-Elitism\n"
            "D. Weak Non-Sadism\n"
            "E. Weak Quality Addition"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: Critical-level views in population axiology subtract a positive "
            "constant from each person's welfare level, then sum to get the population's value.\n"
            "Step 2: Weak Non-Sadism says there is a negative welfare level and a number "
            "of lives at that level such that adding any number of lives with positive "
            "welfare is at least as good as adding the lives with negative welfare.\n"
            "Step 3: On critical-level views, adding lives barely worth living makes a "
            "population non-diminishingly worse, because those lives get negative value "
            "once you subtract the positive constant. Adding enough of them can make "
            "any population arbitrarily bad.\n"
            "Step 4: This directly violates Weak Non-Sadism, since adding positive-welfare "
            "lives can be worse than adding negative-welfare lives.\n"
            "Step 5: The answer is D.{thinking_token_close}\n"
            "Final Answer: D"
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: The concept of logical \"depth\" mentioned in "
            "_The Quark and the Jaguar_ has a reciprocal/inverse concept "
            "(associated with Charles Bennett); take the third letter of that "
            "reciprocal concept word and call it c1.\n"
            "After being admitted to MIT, Murray Gell-Man thought of suicide, "
            "having the ability to (1) try MIT or (2) commit suicide. He joked P"
            "\"the two _ didn't commute.\" Let the third character of the missing "
            "word in the quote be called c2.\n"
            "The GELU's last author's last name ends with this letter; call it c3.\n"
            "Now take that letter and Rot13 it; call that letter c4.\n"
            "Is Mars closer in mass to the Earth or to the Moon? Take the second "
            "letter of the answer to this question and call that c5.\n"
            "Output the concatenation of c1, c2, c4, and c5 "
            "(make all characters lowercase)."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1 (c1): The inverse of logical \"depth\" (associated with Charles Bennett) "
            "is \"crypticity\". The third letter of \"crypticity\" is 'y'. So c1 = y.\n"
            "Step 2 (c2): Murray Gell-Mann joked \"the two operators didn't commute.\" "
            "The third character of \"operators\" is 'e'. So c2 = e.\n"
            "Step 3 (c3): The GELU paper's last author is Dan Hendrycks and Gimpel — "
            "the last author is Gimpel, whose name ends in 'l'. So c3 = l.\n"
            "Step 4 (c4): Rot13 of 'l' is 'y'. So c4 = y.\n"
            "Step 5 (c5): Mars (6.39 x 10^23 kg) is closer in mass to the Moon "
            "(7.34 x 10^22 kg) than to the Earth (5.97 x 10^24 kg). "
            "The second letter of \"Moon\" is 'o'. So c5 = o.\n"
            "Step 6: Concatenating c1, c2, c4, c5 = y, e, y, o = \"yeyo\".{thinking_token_close}\n"
            "Final Answer: yeyo"
        ),
    },
]


# --- BFCL (Berkeley Function Calling Leaderboard) -------------------------
# Synthetic examples — NOT from the real BFCL v1 dataset.
BFCL_FEW_SHOT = [
    # Example 1: simple single function call
    {
        "role": "user",
        "content": (
            "I want to convert 25 degrees Celsius to Fahrenheit.\n"
            "\n"
            "Here is a list of functions in json format that you can invoke.\n"
            '[{"name": "temperature.convert", "description": "Convert a temperature value '
            'from one unit to another.", "parameters": {"type": "object", "properties": '
            '{"value": {"type": "number", "description": "The temperature value to convert."}, '
            '"from_unit": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"], '
            '"description": "The source unit."}, '
            '"to_unit": {"type": "string", "enum": ["celsius", "fahrenheit", "kelvin"], '
            '"description": "The target unit."}}, '
            '"required": ["value", "from_unit", "to_unit"]}}]'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The user wants to convert 25 degrees Celsius to Fahrenheit. "
            "I need to identify the correct function and fill in the parameters.\n"
            "Step 2: The function temperature.convert takes value, from_unit, and to_unit. "
            "value=25, from_unit=\"celsius\", to_unit=\"fahrenheit\".\n"
            "Step 3: All required parameters are provided. I will make a single call.{thinking_token_close}\n"
            "Final Answer: [temperature.convert(value=25, from_unit=\"celsius\", to_unit=\"fahrenheit\")]"
        ),
    },
    # Example 2: parallel calls to two different functions
    {
        "role": "user",
        "content": (
            "Book a flight from New York to London on 2025-08-15 and also reserve a hotel "
            "in London for 3 nights starting 2025-08-15.\n"
            "\n"
            "Here is a list of functions in json format that you can invoke.\n"
            '[{"name": "flights.search", "description": "Search and book a one-way flight.", '
            '"parameters": {"type": "object", "properties": '
            '{"origin": {"type": "string", "description": "Departure city."}, '
            '"destination": {"type": "string", "description": "Arrival city."}, '
            '"date": {"type": "string", "description": "Travel date in YYYY-MM-DD format."}}, '
            '"required": ["origin", "destination", "date"]}}, '
            '{"name": "hotels.reserve", "description": "Reserve a hotel room.", '
            '"parameters": {"type": "object", "properties": '
            '{"city": {"type": "string", "description": "City of the hotel."}, '
            '"check_in": {"type": "string", "description": "Check-in date in YYYY-MM-DD format."}, '
            '"nights": {"type": "integer", "description": "Number of nights to stay."}}, '
            '"required": ["city", "check_in", "nights"]}}]'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "{thinking_token_open}Let's think step-by-step.\n"
            "Step 1: The user has two distinct requests: (a) book a flight NYC to London on 2025-08-15, "
            "and (b) reserve a hotel in London for 3 nights starting 2025-08-15.\n"
            "Step 2: For the flight, I use flights.search with origin=\"New York\", "
            "destination=\"London\", date=\"2025-08-15\".\n"
            "Step 3: For the hotel, I use hotels.reserve with city=\"London\", "
            "check_in=\"2025-08-15\", nights=3.\n"
            "Step 4: These are independent requests, so I should make both calls in parallel.{thinking_token_close}\n"
            "Final Answer: [flights.search(origin=\"New York\", destination=\"London\", "
            "date=\"2025-08-15\"), hotels.reserve(city=\"London\", check_in=\"2025-08-15\", nights=3)]"
        ),
    },
]


FEW_SHOT_PROMPT_REGISTRY: dict[str, dict[str, str]] = {
    "bfcl":                 BFCL_FEW_SHOT,
    "bigbench_movie": BIGBENCH_MOVIE,
    "bigbench_causal": BIGBENCH_CAUSAL,
    "logiqa": LOGIQA_FEW_SHOT_MESSAGES,
    "codeqa": CODEQA_FEW_SHOT,
    "cs1qa":                CS1QA_FEW_SHOT,
    "hotpotqa": HOTPOTQA_FEW_SHOT,
    "college_math_test":    COLLEGE_MATH_FEW_SHOT,
    "olympiadbench":        OLYMPIADBENCH_FEW_SHOT,
    "math500":              MATH500_FEW_SHOT,
    "hle":                  HLE_FEW_SHOT,
}


THINKING_TOKENS: dict[str, tuple[str, str]] = {
    "none": ("", ""),
    "Qwen/Qwen3.5-27B": ("<think>\n", "\n</think>"),
    # Gpt-OSS uses OpenAI Harmony format: https://developers.openai.com/cookbook/articles/openai-harmony/
    # "openai/gpt-oss-20b": ("<|channel|>analysis<|message|>\n", "\n<|end|>")
    "openai/gpt-oss-20b": ("", "")
}




