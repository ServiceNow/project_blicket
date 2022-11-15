import numpy as np
import openai
import os
import scipy

openai_api_key = os.environ["OPENAI_API_KEY"]


def gpt3_call(
    engine="text-ada-001",
    prompt="",
    max_tokens=128,
    temperature=0,
    logprobs=1,
    echo=False,
):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        echo=echo,
    )

    return response


def gpt3_scoring(
    prompt,
    options,
    engine="text-ada-001",
    verbose=False,
    n_tokens_score=9999999999,
    lock_token=None,
):
    verbose and print("Scoring", len(options), "options")
    gpt3_prompt_options = [f"{prompt}{o}" for o in options]
    verbose and print(gpt3_prompt_options[0], "\n" * 4)
    response = gpt3_call(
        engine=engine,
        prompt=gpt3_prompt_options,
        max_tokens=0,
        logprobs=1,
        temperature=0,
        echo=True,
    )
    scores = []
    for option, choice in zip(options, response["choices"]):
        if lock_token is not None:
            n_tokens_score = choice["logprobs"]["tokens"][::-1].index(lock_token)
        tokens = choice["logprobs"]["tokens"][-n_tokens_score:]
        verbose and print("Tokens:", tokens)
        token_logprobs = choice["logprobs"]["token_logprobs"][-n_tokens_score:]
        total_logprob = 0
        denom = 0
        for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
            if token_logprob is not None:
                denom += 1
                total_logprob += token_logprob

        scores.append(total_logprob / denom)

    return np.array(scores)


if __name__ == "__main__":
    prompt = "The machine turns on when both the blue pyramid and the black square is placed on it.\nQuestion: Which object should I place on the machine to make it turn on? Answer: "
    options = [
        "black square",
        "blue cone",
        "blue pyramid",
        "blue pyramid or black square",
        "blue pyramid and black square",
    ]

    log_scores = gpt3_scoring(prompt, options)
    scores = scipy.special.softmax(log_scores)
    print(scores)
