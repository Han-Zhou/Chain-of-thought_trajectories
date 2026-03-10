"""
Few-shot Chain-of-Thought prompts for each supported dataset.
"""



# --- LogiQA ----------------------------------------------------------------
# Sourced from eval.txt at https://github.com/lgw863/LogiQA-dataset (the official github repo for LogiQA)
LOGIQA_FEW_SHOT = """\
------
Context:
Black Americans are twice as likely to suffer from hypertension as white Americans.The same is true when comparing Westernized black Africans to white Africans.The researchers hypothesized that the reason why westernized black people suffer from hypertension is the result of the interaction of two reasons? one is the high salt content of western foods, and the other is the adaptation mechanism of black genetic genes to the salt-deficient environment .

Question:
The following conclusions about contemporary westernized African blacks, if the item is true, can it best support the researchers' hypothesis?

Options:
A.The blood pressure of the descendants of Senegalese and Gambians is usually not high, and the history of Senegal and Gambia has not been short of salt.
B.The unusually high salt intake in certain parts of Africa is a serious problem that threatens the health of residents.
C.Considering health care, most African whites also pay attention to controlling salt intake.
D.The blood pressure of Yoruba people in West Africa is not high.Yoruba people have lived inland far away from sea salt and far away from the Sahara salt mine in Africa.

Let's think step-by-step.
Step 1: From the hypothesis, Hypertension = (Genetic trait to save salt) + (Western high-salt diet).
Step 2: Identify the Necessary Evidence: I need to find a group where the "Genetic trait" is missing to see if the "Hypertension" also disappears.
Step 3: From Option A, Residents of Senegal/Gambia had plenty of salt (no need for the gene). Their descendants in the West have normal blood pressure.
Step 4: This confirms that when the "salt-deficient" genetic history is removed, the Western diet doesn't cause the same level of hypertension.
Step 5: Option A is the best fit.
Final Answer: A

------
Context:
The prohibition of advertising cigarettes on public media does not reduce the number of young people smoking, because they have known for a long time that there are cigarettes in the world, known for various brands of cigarettes, and know where to get them.They do not need advertisements to provide this information.

Question:
The following, if true, can weaken the above argument most?

Options:
A.Watching or listening to advertisements can increase a person's desire to obtain such products.
B.Prohibition of cigarette advertisements on public media will cause other forms of cigarette advertisements to proliferate.
C.Cigarette advertising on public media is a major expense for tobacco companies.
D.Anti-smokers have advertised in the public media against smoking from the beginning.

Let's think step-by-step.
Step 1: The argument claims that ads don't work because young people already know cigarettes exist and where to buy them. It assumes ads only function as a source of "missing information."
Step 2: The argument overlooks that advertising isn't just about providing information; it’s about persuasion and desire.
Step 3: To weaken this, we need an option that shows advertisements do more than just tell people "cigarettes are for sale"—they actually drive the urge to smoke.
Step 4: A directly addresses this by stating ads increase the desire to obtain the product. B and C discuss the consequences for companies, not the behavior of young people. D talks about anti-smoking ads, which doesn't explain why cigarette ads themselves are effective or ineffective.
Step 5: Option A is the best choice to weaken the argument.
Final Answer: A
"""


