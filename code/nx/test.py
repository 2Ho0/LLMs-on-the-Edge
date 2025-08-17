import os
import pandas as pd
import time
import subprocess
import ollama
import threading # 스레딩 라이브러리 추가

measurements = []

# LLM 응답 생성을 별도 스레드에서 처리하는 함수
def run_generation(question, results_dict):
    """
    Ollama 모델에 스트리밍으로 요청하고 TTFT와 전체 응답을 results_dict에 저장합니다.
    """
    try:
        start_time = time.time()
        first_token_time = None
        ttft = None
        full_response = ""
        options = {"stop": ["\n", "\n\n", ".\"", "!\"", "?\""]}
        
        # ollama.chat을 stream=True로 호출하여 응답을 스트리밍
        stream = ollama.chat(
            model='phi3.5',
            messages=[{'role': 'user', 'content': question}],
            options = options,
            stream=True
        )

        for chunk in stream:
            # 첫 번째 응답 조각을 받았을 때 TTFT 계산
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - start_time

            # 응답 내용 이어붙이기
            content = chunk['message']['content']
            full_response += content
        
        # 결과를 공유 딕셔너리에 저장
        results_dict['response'] = full_response
        results_dict['ttft'] = ttft
        
    except Exception as e:
        print(f"Error generating response from Ollama: {e}")
        results_dict['response'] = ""
        results_dict['ttft'] = -1.0

def monitor_tegrastats_during_response(question):
    """
    응답 생성 동안 tegrastats를 모니터링하고 주요 성능 지표를 계산합니다.
    """
    try:
        # tegrastats를 백그라운드 프로세스로 실행
        process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        gpu_utilizations = []
        power_consumptions_mW = [] # 측정된 전력을 와트(W) 단위로 저장
        ram_usages = []
        is_sampling = False

        # 스레드 간 데이터 공유를 위한 딕셔너리
        results = {}
        # LLM 응답 생성을 위한 스레드 생성 및 시작
        gen_thread = threading.Thread(target=run_generation, args=(question, results))
        gen_thread.start()

        # 스레드가 살아있는 동안 tegrastats 출력 모니터링
        while gen_thread.is_alive() or (process.stdout and not process.stdout.closed):
            line = process.stdout.readline()
            if not line:
                if not gen_thread.is_alive():
                    break
                else:
                    continue

            # GPU 사용률이 0보다 커지면 샘플링 시작
            if "GR3D_FREQ" in line:
                try:
                    parts = line.split()
                    gpu_util_str = parts[parts.index("GR3D_FREQ") + 1]
                    gpu_utilization = int(gpu_util_str.replace('%', ''))
                    
                    if gpu_utilization > 0 and not is_sampling:
                        is_sampling = True
                    
                    if is_sampling:
                        gpu_utilizations.append(gpu_utilization)
                except (ValueError, IndexError):
                    pass

            if is_sampling:
                # RAM 사용량 추출
                if "RAM" in line:
                    try:
                        ram_str = line.split("RAM ")[1].split("/")[0]
                        ram_usages.append(int(ram_str))
                    except (ValueError, IndexError):
                        pass
                
                # 전력 소비량 추출 (단위: W)
                if "VDD_IN" in line:
                    try:
                        parts= line.split()
                        vdd_in_index = parts.index("VDD_IN")
                        power_value = parts[vdd_in_index +1].split('mW')[0]
                        print("power_str", power_value)
                        power_consumptions_mW.append(float(power_value))
                    except (ValueError, IndexError):
                        pass

            # 응답 생성이 끝나고(스레드 종료) GPU 사용량이 0으로 돌아오면 루프 종료
            if not gen_thread.is_alive() and is_sampling and gpu_utilizations and gpu_utilizations[-1] == 0:
                break
            
            time.sleep(0.1)

        # 프로세스 및 스레드 정리
        process.terminate()
        gen_thread.join()

        # 마지막 불안정한 값 제거
        print("power_consumptions_mW: ", power_consumptions_mW)
        if power_consumptions_mW: del power_consumptions_mW[-1]
        print("power_consumptions_mW: ", power_consumptions_mW)
        if gpu_utilizations: del gpu_utilizations[-1]
        if ram_usages: del ram_usages[-1]
        
        # 평균값 계산
        avg_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0
        avg_power_consumption_mW = sum(power_consumptions_mW) / len(power_consumptions_mW) if power_consumptions_mW else 0
        avg_ram_usage = sum(ram_usages) / len(ram_usages) if ram_usages else 0

        # 새 지표 추출
        response = results.get('response', "")
        ttft = results.get('ttft', 0)
        response_token_sum = len(response.split())

        # 총에너지 소비량(mWh) 및 토큰당 에너지 소비량(J/token) 계산
        duration_s = len(power_consumptions_mW) * 0.1 # 측정 간격을 0.1초로 가정
        total_energy_mWh = (avg_power_consumption_mW * duration_s) / 3600.0 if duration_s > 0 else 0
        total_energy_J = total_energy_mWh * 3.6
        energy_per_token_J = total_energy_J / response_token_sum if response_token_sum > 0 else 0

        return response, avg_gpu_utilization, total_energy_mWh, avg_ram_usage, ttft, energy_per_token_J

    except Exception as e:
        print(f"Error monitoring tegrastats: {e}")
        return None, 0, 0, 0, 0, 0

def readQuestions():
    # TriviaQA.csv 파일이 코드와 다른 폴더에 있다면 경로를 수정해주세요.
    # 예: '../../dataset/TriviaQA.csv'
    df = pd.read_csv('../../dataset/TriviaQA.csv', sep=',')
    return df

def extractDataFromResponse(response, question, reference_answer, latency, total_energy_mWh, gpu_utilization, ram_usage, ttft, energy_per_token):
    generated_answer = response.replace(';', '')
    response_token_sum = len(generated_answer.split())
    prompt_token_sum = len(question.split())
    ref_answer_token_sum = len(reference_answer.split())

    print(f"question: {question}")
    print(f"reference_answer: {reference_answer}")
    print(f"generated_answer: {generated_answer}")
    print(f"number of tokens in the response: {response_token_sum}")
    print(f"number of tokens in the prompt: {prompt_token_sum}")
    print(f"number of tokens in the reference answer: {ref_answer_token_sum}")
    print(f"Latency (s): {latency:.4f}")
    print(f"Time to First Token (s): {ttft:.4f}")
    print(f"Total Energy Consumption (mWh): {total_energy_mWh:.4f}") # mWh 단위로 출력
    print(f"Average GPU Utilization (%): {gpu_utilization:.2f}")
    print(f"Average RAM Usage (MB): {ram_usage:.2f}")
    print(f"Energy per Token (J/token): {energy_per_token:.4f}")
    print("---------------------------------------")

    measurements.append([question, reference_answer, generated_answer,
                         response_token_sum, prompt_token_sum, ref_answer_token_sum,
                         latency, ttft, total_energy_mWh, gpu_utilization, ram_usage, energy_per_token])

def askQuestions():
    df = readQuestions()
    for index, row in df.iterrows():
        print(f"Processing question index: {index}")
        question = row['question']
        reference_answer = row['answer']

        start_time = time.time()

        response, gpu_utilization, total_energy_mWh, ram_usage, ttft, energy_per_token = monitor_tegrastats_during_response(question)

        end_time = time.time()
        latency = end_time - start_time

        if response is not None:
            extractDataFromResponse(response, question, reference_answer, latency, total_energy_mWh, gpu_utilization, ram_usage, ttft, energy_per_token)
        
        time.sleep(1)

    # 결과를 DataFrame으로 저장
    columns = ['question', 'reference_answer', 'generated_answer',
               'response_token_sum', 'prompt_token_sum', 'ref_answer_token_sum',
               'latency_s', 'ttft_s', 'total_energy_mWh', # 컬럼명 변경
               'gpu_utilization_percent', 'ram_usage_MB', 'energy_per_token_J']
    df_results = pd.DataFrame(measurements, columns=columns)
    
    print(df_results)
    df_results.to_excel('results_phi35_with_final_metrics.xlsx', index=False)

# 스크립트 실행
if __name__ == "__main__":
    askQuestions()
