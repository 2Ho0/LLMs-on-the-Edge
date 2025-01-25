import os
import pandas as pd
import time
import subprocess
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

measurements = []

# tegrastats를 실행하여 GPU 및 전력 소비 정보 추출
def get_tegrastats_metrics():
    try:
        # tegrastats 실행
        process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 1초 동안 출력 수집
        output_lines = []
        start_time = time.time()
        while time.time() - start_time < 1:  # 1초 동안 출력 수집
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())

        # 프로세스 종료
        process.terminate()

        # 출력이 없으면 None 반환
        if not output_lines:
            print("No output from tegrastats")
            return None, None

        # 마지막 줄에서 필요한 정보 추출
        last_line = output_lines[-1]

        # GPU 사용률 추출 (GR3D_FREQ)
        gpu_utilization = None
        if "GR3D_FREQ" in last_line:
            try:
                # "GR3D_FREQ" 위치를 찾고 그 다음 값을 추출
                parts = last_line.split()
                gr3d_index = parts.index("GR3D_FREQ")
                gpu_utilization = int(parts[gr3d_index + 1].replace('%', ''))
            except (IndexError, ValueError):
                print("Error parsing GR3D_FREQ")

        # 전력 소비 추출 (VDD_IN)
        power_consumption = None
        if "VDD_IN" in last_line:
            try:
                # "VDD_IN" 위치를 찾고 그 다음 값을 추출
                parts = last_line.split()
                vdd_in_index = parts.index("VDD_IN")
                power_value = parts[vdd_in_index + 1].split('mW')[0]  # "7894mW"에서 숫자만 추출
                power_consumption = float(power_value) / 1000  # mW를 W로 변환
            except (IndexError, ValueError):
                print("Error parsing VDD_IN")

        return gpu_utilization, power_consumption

    except Exception as e:
        print("Error measuring tegrastats metrics:", e)
        return None, None



# Load questions from the dataset
def readQuestions():
    current_directory = os.getcwd()
    df = pd.read_csv('../dataset/miniSQuAD.csv', sep=';')
    return df

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer, device=0)

# Ask the Hugging Face model a question and receive a response
def generateResponse(question, context):
    response = qa_pipeline(question=question, context=context)
    return response

# Extract data from the response to store it in a dataframe
def extractDataFromResponse(response, question, reference_answer, latency, power_consumption, gpu_utilization):

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
    print("Latency (s):", latency)
    print("Power consumption (W):", power_consumption)
    print("GPU Utilization (%):", gpu_utilization)
    print("---------------------------------------")

    # Append the values to a list to save them later
    measurements.append([question, reference_answer, generated_answer, confidence,
                         response_token_sum, prompt_token_sum, ref_answer_token_sum,
                         latency, power_consumption, gpu_utilization])


# Ask every question in the dataset
def askQuestions():
    df = readQuestions()
    for index, row in df.iterrows():
        print(index)
        question = row['question']
        reference_answer = row['answer']
        context = reference_answer  # Context is the reference answer in this case

        # Measure start time
        start_time = time.time()
        gpu_before_utilization, power_before = get_tegrastats_metrics()

        # Generate response
        response = generateResponse(question, context)

        # Measure end time
        end_time = time.time()
        gpu_after_utilization, power_after = get_tegrastats_metrics()

        # Calculate metrics
        latency = end_time - start_time
        if power_before is not None and power_after is not None:
            avg_power = (power_before + power_after) / 2
        else:
            avg_power = None

        if gpu_before_utilization is not None and gpu_after_utilization is not None:
            avg_gpu_utilization = (gpu_before_utilization + gpu_after_utilization) / 2
        else:
            avg_gpu_utilization = None

        # Extract data
        extractDataFromResponse(response, question, reference_answer, latency, avg_power, avg_gpu_utilization)
        time.sleep(1)

    # Save the response
    df = pd.DataFrame(measurements,
                      columns=['question', 'reference_answer', 'generated_answer',
                               'confidence_score', 'response_token_sum', 'prompt_token_sum',
                               'ref_answer_token_sum', 'latency', 'power_consumption',
                               'gpu_utilization'])
    print(df)
    df.to_excel('results_roberta_base_squad2_with_tegrastats.xlsx')
    df.to_csv('results_roberta_base_squad2_with_tegrastats.csv', sep=';')

# Execute the function
askQuestions()
