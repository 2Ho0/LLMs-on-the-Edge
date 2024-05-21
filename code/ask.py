import os
import pandas as pd
import ollama
from transformers import AutoTokenizer
import time

measurements = []
# Load questions from the dataset
def readQuestions():
  current_directory = os.getcwd()
  df = pd.read_csv('dataset/miniSQuAD.csv', sep=';')
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
def extractDataFromResponse(response, question, reference_answer):

    #LLM-generated answer 
    generated_answer = response['message']['content'].replace(';', '')
    #time spent in seconds loading the model
    load_duration = response['load_duration'] / 1000000000
    #time spent generating the response
    total_duration = response['total_duration'] / 1000000000
    #time spent in seconds evaluating the prompt
    prompt_eval_duration = response['prompt_eval_duration'] / 1000000000
    #time in seconds spent generating the response
    eval_duration = response['eval_duration'] / 1000000000
    #number of tokens in the response
    response_token_sum = response['eval_count']
    #number of tokens in the prompt
    tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    prompt_token_sum = len(tokenizer(question)["input_ids"])
    #number of tokens in the reference answer of the dataset
    ref_answer_token_sum = len(tokenizer(reference_answer)["input_ids"])

    #Print everything in the terminal
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
    print("---------------------------------------")

    #Append the values to a list to save them later
    measurements.append([question, reference_answer, generated_answer, 
                         load_duration, total_duration, prompt_eval_duration,
                         eval_duration, response_token_sum, prompt_token_sum, 
                         ref_answer_token_sum])
    

# Ask every question in the dataset
def askQuestions(model):
    df = readQuestions()
    for index, row in df.iterrows():
        print(index)
        question = 'Answer very briefly: ' + row['question']
        reference_answer = row['answer']
        response = generateResponse(question, model)
        extractDataFromResponse(response, question, reference_answer)
        time.sleep(3)

    #Safe the response
    df = pd.DataFrame(measurements, 
                  columns = ['question', 'reference_answer', 'generated_answer', 
                             'load_duration', 'total_duration', 'prompt_eval_duration',
                             'eval_duration', 'response_token_sum', 'prompt_token_sum',
                             'ref_answer_token_sum']) 
    print(df)
    df.to_excel('results_' + model + '_PI5_brief.xlsx')
    df.to_csv('results_' + model + '_PI5_brief.csv', sep=';')

#Change the name of the model here before execution
askQuestions('phi3')
