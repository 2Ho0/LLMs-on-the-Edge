import os
import pandas as pd
import time
import subprocess
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

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

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Ask the Hugging Face model a question and receive a response
def generateResponse(question, context):
    response = qa_pipeline(question=question, context=context)
    return response

# Extract data from the response to store it in a dataframe
def extractDataFromResponse(response, question, reference_answer, power_consumption):

    # LLM-generated answer
    generated_answer = response['answer']
    # Confidence score for the generated answer
    confidence = response['score']
    # Number of tokens in the response
    response_token_sum = len(tokenizer(generated_answer)["input_ids"])
    # Number of tokens in the prompt
    prompt_token_sum = len(tokenizer(question)["input_ids"])
    # Number of tokens in the reference answer
    ref_answer_token_sum = len(tokenizer(reference_answer)["input_ids"])

    # Print everything in the terminal
    print("question: ", question)
    print("reference_answer: ", reference_answer)
    print("generated_answer: ", generated_answer)
    print("confidence_score: ", confidence)
    print("number of tokens in the response: ", response_token_sum)
    print("number of tokens in the prompt: ", prompt_token_sum)
    print("number of tokens in the reference answer: ", ref_answer_token_sum)
    print("Power consumption (W):", power_consumption)
    print("---------------------------------------")

    # Append the values to a list to save them later
    measurements.append([question, reference_answer, generated_answer, confidence,
                         response_token_sum, prompt_token_sum, ref_answer_token_sum, power_consumption])


# Ask every question in the dataset
def askQuestions():
    df = readQuestions()
    for index, row in df.iterrows():
        print(index)
        question = row['question']
        reference_answer = row['answer']
        context = reference_answer  # Context is the reference answer in this case

        # Measure power usage before and after generating the response
        power_before = get_power_usage()
        response = generateResponse(question, context)
        power_after = get_power_usage()

        # Calculate the average power consumption
        if power_before is not None and power_after is not None:
            avg_power = (power_before + power_after) / 2
        else:
            avg_power = None

        extractDataFromResponse(response, question, reference_answer, avg_power)
        time.sleep(1)

    # Save the response
    df = pd.DataFrame(measurements,
                      columns=['question', 'reference_answer', 'generated_answer',
                               'confidence_score', 'response_token_sum', 'prompt_token_sum',
                               'ref_answer_token_sum', 'power_consumption'])
    print(df)
    df.to_excel('results_roberta_base_squad2_with_power.xlsx')
    df.to_csv('results_roberta_base_squad2_with_power.csv', sep=';')

# Execute the function
askQuestions()
