import os
import pandas as pd
import time
import subprocess
import ollama

measurements = []

def monitor_tegrastats_during_response(question):
    try:
        # tegrastats 실행
        process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        gpu_utilizations = []
        power_consumptions = []
        is_sampling = False  # 샘플링 시작 여부

        # 응답 생성 시작
        response = None

        while True:
            line = process.stdout.readline()
            if line:
                # GPU 사용률 및 전력 소비 추출
                if "GR3D_FREQ" in line:
                    try:
                        parts = line.split()
                        gr3d_index = parts.index("GR3D_FREQ")
                        gpu_utilization = int(parts[gr3d_index + 1].replace('%', ''))

                        # GPU 사용률이 0%에서 벗어나면 샘플링 시작
                        if gpu_utilization > 0 and not is_sampling:
                            is_sampling = True

                        # 샘플링이 시작되었으면 데이터를 수집
                        if is_sampling:
                            gpu_utilizations.append(gpu_utilization)
                    except (IndexError, ValueError):
                        print("Error parsing GR3D_FREQ")

                if "VDD_IN" in line and is_sampling:
                    try:
                        parts = line.split()
                        vdd_in_index = parts.index("VDD_IN")
                        power_value = parts[vdd_in_index + 1].split('mW')[0]
                        power_consumption = float(power_value) / 1000  # mW를 W로 변환
                        power_consumptions.append(power_consumption)
                    except (IndexError, ValueError):
                        print("Error parsing VDD_IN")

            # 응답 생성 시작(한 번만 호출)
            if response is None:
                response = generateResponse(question)

            # GPU 사용률이 다시 0%가 되면 루프 종료
            if is_sampling and gpu_utilizations and gpu_utilizations[-1] == 0:
                break

            # CPU 사용률을 줄이기 위해 약간의 지연 추가
            time.sleep(0.1)

        # 프로세스 종료
        process.terminate()
        del power_consumptions[-1]
        del gpu_utilizations[-1]
        
        print(power_consumptions)
        print(gpu_utilizations)

        # 평균값 계산
        avg_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else None
        avg_power_consumption = sum(power_consumptions) / len(power_consumptions) if power_consumptions else None

        return response, avg_gpu_utilization, avg_power_consumption

    except Exception as e:
        print("Error monitoring tegrastats during response:", e)
        return None, None, None

# Load questions from the dataset
def readQuestions():
    current_directory = os.getcwd()
    df = pd.read_csv('../dataset/mini_dataset.csv', sep=',')
    return df

# Ask the Ollama model a question and receive a response
def generateResponse(question):
    try:
        response = ollama.chat(
            model='deepseek-r1:8b',
            messages=[
                {
                    'role': 'user',
                    'content': question,
                    'stream': 'false'
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        print("Error generating response from Ollama:", e)
        return ""

# Extract data from the response to store it in a dataframe
def extractDataFromResponse(response, question, reference_answer, latency, power_consumption, gpu_utilization):

    # LLM-generated answer
    generated_answer = response.replace(';', '')
    # Number of tokens in the response
    response_token_sum = len(generated_answer.split())
    # Number of tokens in the prompt
    prompt_token_sum = len(question.split())
    # Number of tokens in the reference answer
    ref_answer_token_sum = len(reference_answer.split())

    # Print everything in the terminal
    print("question: ", question)
    print("reference_answer: ", reference_answer)
    print("generated_answer: ", generated_answer)
    print("number of tokens in the response: ", response_token_sum)
    print("number of tokens in the prompt: ", prompt_token_sum)
    print("number of tokens in the reference answer: ", ref_answer_token_sum)
    print("Latency (s):", latency)
    print("Average Power consumption (W):", power_consumption)
    print("Average GPU Utilization (%):", gpu_utilization)
    print("---------------------------------------")

    # Append the values to a list to save them later
    measurements.append([question, reference_answer, generated_answer,
                         response_token_sum, prompt_token_sum, ref_answer_token_sum,
                         latency, power_consumption, gpu_utilization])

# Ask every question in the dataset
def askQuestions():
    df = readQuestions()
    for index, row in df.iterrows():
        print(index)
        question = row['question']
        reference_answer = row['answer']

        # Measure start time
        start_time = time.time()

        # Monitor tegrastats and generate response
        response, gpu_utilization, power_consumption = monitor_tegrastats_during_response(question)

        # Measure end time
        end_time = time.time()

        # Calculate latency
        latency = end_time - start_time

        # Extract data
        extractDataFromResponse(response, question, reference_answer, latency, power_consumption, gpu_utilization)
        time.sleep(1)

    # Save the response
    df = pd.DataFrame(measurements,
                      columns=['question', 'reference_answer', 'generated_answer',
                               'response_token_sum', 'prompt_token_sum',
                               'ref_answer_token_sum', 'latency', 'power_consumption',
                               'gpu_utilization'])
    print(df)
    df.to_excel('results_deepseek_with_tegrastats.xlsx')
    df.to_csv('results_deepseek_with_tegrastats.csv', sep=';')

# Execute the function
askQuestions()

