import os
import pandas as pd
import ollama
from transformers import AutoTokenizer
import time
import subprocess

measurements = []

# GPU 전력 소비 측정 함수
def get_power_usage():
    try:
        # Execute nvidia-smi command to get power usage
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        power = float(result.stdout.strip().split('\n')[0])  # 첫 번째 GPU 전력 소비 반환
        return power
    except Exception as e:
        print("Error measuring power usage:", e)
        return None

# Load questions from the dataset
def readQuestions():
    current_directory = os.getcwd()
    df = pd.read_csv('../dataset/miniSQuAD.csv', sep=';')
    return df

# Ask the LLM a question and receive a response
def generateResponse(question, llm):
    response = ollama.chat(model=llm, messages=[
        {
            'role': 'user',
            'content': question,
            'stream': 'false'
        },
    ])
    return response

# Extract data from the response to store it in a dataframe
def extractDataFromResponse(response, question, reference_answer, power_consumption):

    # LLM-generated answer
    generated_answer = response['message']['content'].replace(';', '')
    # time spent in seconds loading the model
    load_duration = response['load_duration'] / 1000000000
    # time spent generating the response
    total_duration = response['total_duration'] / 1000000000
    # time spent in seconds evaluating the prompt
    prompt_eval_duration = response['prompt_eval_duration'] / 1000000000
    # time in seconds spent generating the response
    eval_duration = response['eval_duration'] / 1000000000
    # number of tokens in the response
    response_token_sum = response['eval_count']
    # number of tokens in the prompt
    tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    prompt_token_sum = len(tokenizer(question)["input_ids"])
    # number of tokens in the reference answer of the dataset
    ref_answer_token_sum = len(tokenizer(reference_answer)["input_ids"])

    # Print everything in the terminal
    print("question: ", question)
    print("reference_answer: ", reference_answer)
    print("total: ", total_duration)
    print("loading the model: ", load_duration)
    print("evaluating the prompt: ", prompt_eval_duration)
    print("generating the response: ", eval_duration)
    print("number of tokens in the response: ", response_token_sum)
    print("number of tokens in the prompt: ", prompt_token_sum)
    print("number of tokens in the reference answer: ", ref_answer_token_sum)
    print("LLM-generated answer: ", generated_answer)
    print("Power consumption (W):", power_consumption)
    print("---------------------------------------")

    # Append the values to a list to save them later
    measurements.append([question, reference_answer, generated_answer,
                         load_duration, total_duration, prompt_eval_duration,
                         eval_duration, response_token_sum, prompt_token_sum,
                         ref_answer_token_sum, power_consumption])


# Ask every question in the dataset
def askQuestions(model):
    df = readQuestions()
    for index, row in df.iterrows():
        print(index)
        question = row['question']
        reference_answer = row['answer']

        # Measure power usage before and after generating the response
        power_before = get_power_usage()
        response = generateResponse(question, model)
        power_after = get_power_usage()

        # Calculate the average power consumption
        if power_before is not None and power_after is not None:
            avg_power = (power_before + power_after) / 2
        else:
            avg_power = None

        extractDataFromResponse(response, question, reference_answer, avg_power)
        time.sleep(3)

    # Save the response
    df = pd.DataFrame(measurements,
                      columns=['question', 'reference_answer', 'generated_answer',
                               'load_duration', 'total_duration', 'prompt_eval_duration',
                               'eval_duration', 'response_token_sum', 'prompt_token_sum',
                               'ref_answer_token_sum', 'power_consumption'])
    print(df)
    df.to_excel('results_' + model + '_PI5_with_power.xlsx')
    df.to_csv('results_' + model + '_PI5_with_power.csv', sep=';')

# Change the name of the model here before execution
askQuestions('phi3')
