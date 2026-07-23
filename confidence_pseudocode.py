# Confidence pseudocode



# Vanilla sample
# T0
reference_cot_with_answer, reference_cot, reference_answer, reference_cot_with_answer_prob, reference_cot_with_answer_cache, question_cache = generate_cots(question, nb=1, temperature=0)
# T1
confidence = confidence_from_answerprob(question, reference_cot_with_answer, reference_cot_with_answer_prob, reference_cot_with_answer_cache)
vanilla_sample = {"cot": reference_cot, "answer": reference_answer, "confidence": confidence}
# T2
vanilla_timings = {"generation_time": T1-T0, "confidence_time": T2-T1}

# Rejection samples
# T0
cot_with_answers, cots, answers, cot_with_answer_caches, cot_with_answer_probs, _ = generate_cots(question, nb=32, temperature=0.9, question_cache)
# T1
for i in range(32):
    if answers[i]==reference_answer:
        confidence = confidence_from_answerprob(question, cot_with_answers[i], cot_with_answer_probs[i], cot_with_answer_caches[i])
        rejection_samples += [{"cot": cots[i], "answer": reference_answer, "confidence": confidence}]
# T2
rejection_timings = {"generation_time": T1-T0, "confidence_time": T2-T1}

# Laywer samples
# T0
cots = alternative_cot_lawyer(question, reference_answer, nb=32, temperature=0.9, question_cache)
# T1
for i in range(32):
    confidence = confidence_from_cot_string(cots[i], reference_answer, question_cache)
    if confidence is not None:
        lawyer_samples += [{"cot": cots[i], "answer": reference_answer, "confidence": confidence}]
# T2
lawyer_timings = {"generation_time": T1-T0, "confidence_time": T2-T1}

# StepBootstrap samples
# T0
cots = alternative_cot_stepbootstrap(question, reference_cot, nb=32)
# T1
for i in range(32):
    confidence = confidence_from_cot_string(cots[i], reference_answer, question_cache)
    if confidence is not None:
        stepbootstrap_samples += [{"cot": cots[i], "answer": reference_answer, "confidence": confidence}]
# T2
stepbootstrap_timings = {"generation_time": T1-T0, "confidence_time": T2-T1}


def generate_cots(question, nb, temperature, question_cache=None):
    if use_api:
        cot_with_answers, cot_with_answer_probs = api.generate(question, nb, temperature)
        cot_with_answer_caches = None
        question_cache = None
    else:
        cot_with_answers, cot_with_answer_probs, cot_with_answer_caches = model.generate(question, nb, temperature, cache=question_cache)
        question_cache = cot_with_answer_caches[0][:question_len]
    
    for i in range(nb):
        cots[i] = extract_cots(cot_with_answers[i])
        answers[i] = extract_answer(cot_with_answers[i])

    if nb>1:
        return cot_with_answers, cots, answers, cot_with_answer_caches, cot_with_answer_probs, question_cache
    else:
        return cot_with_answers[0], cots[0], answers[0], cot_with_answer_caches[0], cot_with_answer_probs[0], question_cache


def confidence_from_answerprob(question, cot_with_answer, cots_with_answer_prob, cot_with_answer_cache)
    confidences[i]['answer_token_probs'] = extract_atps(cots_with_answer_prob)
    if use_api:
        _, confidences[i]['indirect_probs'] = api.generate(question+cot_with_answer+indirect, nb=1, temperature=0)
        _, confidences[i]['verbal_probs'] = api.generate(question+cot_with_answer+verbal, nb=1, temperature=0)
    else:
        _, confidences[i]['indirect_probs'] = model.forward(question+cot_with_answer+indirect, cache=cot_with_answers_cache)
        _, confidences[i]['verbal_probs'] = model.forward(question+cot_with_answer+verbal, cache=cot_with_answer_cache)
    
    return confidence


def alternative_cot_lawyer(question, answer, nb, temperature, question_cache=None)
    if use_api:
        cot_with_answers, _ = api.generate(question+answer_is(answer), nb, temperature)
    else:
        cot_with_answers, _ = model.generate(question+answer_is(answer), nb, temperature, cache=question_cache)
        
    for i in range(nb):
        lawyer_cots[i] = extract_cots(cot_with_answers[i])
    
    return lawyer_cots


def alternative_cot_stepbootstrap(question, reference_cot, nb)
    for i in range(nb):
        steps = split_steps(reference_cot)
        resampled_steps = np.random.choice(steps[:-1], size=len(steps)-1)
        stepbootstrap_cots += [rebuild_cot(*resampled_steps, steps[-1])]
    
    return stepbootstrap_cots


def confidence_from_cot_string(question, cot, reference_answer, question_cache=None):
    if use_api:
        end, end_probs = api.generate(question+cot+"\nThe answer is \\boxed{", nb=1, temperature=0)
        if extract_answer(end)==reference_answer:
            confidence['answer_token_probs'] = extract_atps(end_probs)
            _, confidence['indirect_probs'] = api.generate(question+cot+"\nThe answer is \\boxed{"+end+indirect, nb=1, temperature=0)
            _, confidence['verbal_probs'] = api.generate(question+cot+"\nThe answer is \\boxed{"+end+verbal, nb=1, temperature=0)
        else:
            # Bad cot
            confidence = None
    else:
        cot_with_answer, cot_with_answer_prob, cot_with_answer_cache = model.forward(question+cot+"\nThe answer is \\boxed{"+reference_answer+"}.", cache=question_cache)
        confidence['answer_token_probs'] = extract_atps(cot_with_answer_prob)
        _, confidence['indirect_probs'] = model.forward(question+cot_with_answer+indirect, cache=cot_with_answer_cache)
        _, confidence['verbal_probs'] = model.forward(question+cot_with_answer+verbal, cache=cot_with_answer_cache)
    
    return confidence
