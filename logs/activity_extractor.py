import pm4py
from collections import Counter
import warnings
import json


LOG_PATH = "sepsis_cases_log.xes"

# Minimum percentage of traces/cases where activity must appear
MIN_TRACE_PERCENT = 10.0


def main():
    warnings.filterwarnings("ignore")

    log = pm4py.read_xes(LOG_PATH, show_progress_bar=False)

    activity_trace_counter = Counter()
    total_traces = 0
    total_events = 0

    # PM4Py loaded log as pandas DataFrame
    if hasattr(log, "columns"):
        if "concept:name" not in log.columns:
            print("No column 'concept:name' found.")
            return

        if "case:concept:name" not in log.columns:
            print("No column 'case:concept:name' found.")
            return

        total_events = len(log)
        total_traces = log["case:concept:name"].nunique()

        # For each trace/case, count activity only once
        for case_id, case_df in log.groupby("case:concept:name"):
            activities_in_trace = set(case_df["concept:name"].dropna())

            for activity in activities_in_trace:
                activity_trace_counter[activity] += 1

    # PM4Py loaded log as classic EventLog
    else:
        total_traces = len(log)

        for trace in log:
            activities_in_trace = set()

            for event in trace:
                activity = event["concept:name"] if "concept:name" in event else None
                if activity is not None:
                    activities_in_trace.add(activity)
                    total_events += 1

            for activity in activities_in_trace:
                activity_trace_counter[activity] += 1

    if total_traces == 0:
        print("No traces found.")
        return

    filtered_activities = {
        activity: trace_count
        for activity, trace_count in sorted(activity_trace_counter.items())
        if (trace_count / total_traces) * 100 >= MIN_TRACE_PERCENT
    }

    activity_list = list(filtered_activities.keys())

    print(f"Total traces/cases: {total_traces}")
    print(f"Total events: {total_events}")
    print(f"Unique activities: {len(activity_trace_counter)}")
    print(f"Activities above trace threshold ({MIN_TRACE_PERCENT}%): {len(filtered_activities)}")

    print("\nActivities by trace occurrence:")
    for activity, trace_count in filtered_activities.items():
        percent = (trace_count / total_traces) * 100
        print(f'"{activity}": appears in {trace_count}/{total_traces} traces, {percent:.2f}%')

    print("\nActivity list:")
    print(json.dumps(activity_list, indent=4))


if __name__ == "__main__":
    main()