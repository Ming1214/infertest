import asyncio
import httpx
import argparse
import time
import orjson
import statistics
import datetime
import os
import numpy as np
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--backend', choices=['openai', 'vllm', 'tgi'])
parser.add_argument('--endpoint', nargs='+')
parser.add_argument('--save-path', type=str)
parser.add_argument('--batch', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
parser.add_argument('--batch-bias', type=float, default=10)
parser.add_argument('--duration', type=int, default=300)
parser.add_argument('--prompt-len', type=int, default=2048)
parser.add_argument('--prompt-delta', type=int, default=80)
parser.add_argument('--max-tokens', type=int, default=128)
parser.add_argument('--session-turns', type=int, default=5)
parser.add_argument('--session-interval', type=int, default=5)
parser.add_argument('--request-interval', type=int, default=5)
parser.add_argument('--use-random', action='store_true')
parser.add_argument('--use-warmup', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


def get_endpoint(endpoint):
    if args.backend in ["openai"]:
        return endpoint + '/v1/completions'
    if args.backend in ['vllm']:
        return endpoint + '/generate'
    if args.backend in ['tgi']:
        return endpoint + '/generate_stream'


def get_body(session_prefix, req_id):
    raw = "0123456789"
    prompt_len = args.prompt_len + req_id*args.prompt_delta
    prompt = session_prefix+raw*(prompt_len//len(raw))+raw[: args.prompt_len%len(raw)]
    if args.backend == 'openai':
        return {
            "prompt": prompt, 
            "model": args.model, 
            "max_tokens": args.max_tokens if req_id > 0 else 1, 
            "temperature": 0, 
            "stream": True, 
            "ignore_eos": True, 
            "stop": []
        }
    if args.backend == 'vllm':
        return {
            "prompt": prompt,
            "max_tokens": args.max_tokens if req_id > 0 else 1,
            "temperature": 0,
            "stream": True,
            "stop": []
        }
    if args.backend == 'tgi':
        return {
            'inputs': prompt,
            'parameters': {
                'do_sample': False,
                'max_new_tokens': args.max_tokens if req_id > 0 else 1,
                'return_full_text': True,
                'stop': []
            }
        }


async def requests_worker(endpoint: str, batch_id: int):
    #await asyncio.sleep(batch_id*args.batch_bias)
    if args.use_random:
        await asyncio.sleep(np.random.exponential(args.batch_bias))
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=None, pool=None, write=None, read=None)) as client:
        endpoint = get_endpoint(endpoint)
        tokens = []
        ticks = []
        
        session_id = -1
        session_first_token_latencies = []
        session_continue_token_latencies = []
        while True:
            if session_id >= 0:
                wait = args.session_interval
                if args.use_random:
                    wait = np.random.exponential(wait)
                await asyncio.sleep(wait)
            session_id += 1
            session_prefix = str(uuid.uuid1())
            num_turn = args.session_turns
            if args.use_random:
                num_turn = np.random.poisson(num_turn)
            request_id = -1 if args.use_warmup else 0
            while request_id < num_turn:
                if request_id >= int(not args.use_warmup):
                    wait = args.request_interval
                    if args.use_random:
                        wait = np.random.exponential(wait)
                    await asyncio.sleep(wait)
                request_id += 1
                body = get_body(session_prefix, request_id)
                ticks.append([time.perf_counter()])
                tokens.append(0)
                sys.stdout.write('.')
                sys.stdout.flush()
                
                async with client.stream('POST', endpoint, json=body) as r:
                    chunk_id = -1
                    async for chunk in r.aiter_text():
                        chunk_id += 1
                        tt = time.perf_counter()
                        if request_id > 0:
                            if chunk_id == 0:
                                if request_id == 1:
                                    session_first_token_latencies.append(tt-ticks[-1][-1])
                                else:
                                    session_continue_token_latencies.append(tt-ticks[-1][-1])
                            ticks[-1].append(tt)
                            tokens[-1] += 1
                        delta = tt - start
                        if delta > args.duration:
                            return ticks, tokens, session_first_token_latencies, session_continue_token_latencies
                        if args.debug:
                            print(batch_id, session_id, request_id, chunk_id, repr(chunk))

async def batch_worker(batches, endpoint):
    result = []
    for batch_no, batch in enumerate(batches):
        print(f'--- process with batch size {batch}')
        workers = []
        for batch_id in range(batch):
            workers.append(requests_worker(endpoint, batch_id))
        total_tokens = 0
        session_first_token_latencies = []
        session_continue_token_latencies = []
        first_token_latencies = []
        non_first_token_latency = []
        all_ticks = []
        for tick_group, tokens, sft_latencies, sct_latencies in await asyncio.gather(*workers):
            session_first_token_latencies.extend(sft_latencies)
            session_continue_token_latencies.extend(sct_latencies)
            total_tokens += sum(tokens)
            all_ticks.append(tick_group)
            for ticks in tick_group:
                if len(ticks) == 1:
                    continue
                diff = np.diff(ticks).tolist()
                first_token_latencies.append(diff[0])
                non_first_token_latency.extend(diff[1:])
                
        result.append({
            'batch': batch,
            'total_token': total_tokens,
            'token_per_s': total_tokens / args.duration,
            'avg_session_first_token_latency': statistics.mean(session_first_token_latencies), 
            'avg_session_continue_token_latency': statistics.mean(session_continue_token_latencies), 
            'avg_first_token_latency': statistics.mean(first_token_latencies),
            'min_token_latency': min(non_first_token_latency),
            'max_token_latency': max(non_first_token_latency),
            'median_token_latency': statistics.median(non_first_token_latency),
            'avg_token_latency': statistics.mean(non_first_token_latency),
        })
        print()
        print(result[-1])
        result[-1]['raw_ticks'] = all_ticks
        if batch_no < len(args.batch) - 1:
            await asyncio.sleep(5)
    return [endpoint, result]

def run_batch_worker(batches, endpoint, i):
    #random.shuffle(batches)
    time.sleep(i)
    return asyncio.run(batch_worker(batches, endpoint))

def main():
    print(args)
    batches = sorted(args.batch)
    final_result = {}
    with ProcessPoolExecutor(max_workers=len(args.endpoint)) as pool:
        futures = [pool.submit(run_batch_worker, batches, endpoint, i) for i, endpoint in enumerate(args.endpoint)]
        for p in as_completed(futures):
            endpoint, result = p.result()
            final_result[endpoint] = result

    now_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    save_path = f'./results/{now_str}_{args.model}_{args.backend}.json'
    with open(args.save_path, 'wb') as fp:
        fp.write(orjson.dumps({
            'model': args.model,
            'backend': args.backend,
            "time": now_str,
            "results": final_result,
            "duration": args.duration, 
            "prompt_len": args.prompt_len, 
            "max_tokens": args.max_tokens
        }))

if __name__ == '__main__':
    main()
