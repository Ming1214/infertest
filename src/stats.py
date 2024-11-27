import fire
import json
from prettytable import PrettyTable


def main(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    titles = ['batch', 'total_token', 'token_per_s', 'avg_session_first_token_latency', 'avg_session_continue_token_latency', 'avg_first_token_latency', 'median_token_latency', 'avg_token_latency']
    results = data["results"]
    if isinstance(results, list):
        results = {"fake_endpoint": results}
    for endpoint, result in results.items():
        tab = PrettyTable(titles)
        rows = []
        for res in result:
            row = []
            for key in titles:
                val = res[key]
                row.append(val)
            rows.append(row)
        rows.sort()
        tab.add_rows(rows)
        tab.float_format = ".3"
        tab.align = "r"
        print("*"*100)
        print(endpoint)
        print("-"*100)
        print(tab)
        print("*"*100)


if __name__ == "__main__":
    fire.Fire(main)


