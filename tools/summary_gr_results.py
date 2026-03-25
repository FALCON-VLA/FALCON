import os
import re
import argparse
import sys


def extract_success_rates(file_path):
    success_rates = []
    try:
        if not os.path.exists(file_path):
            print(f"Error: Log file not found: {file_path}", file=sys.stderr)
            return None

        if os.path.getsize(file_path) == 0:
            print(f"Error: Log file is empty: {file_path}", file=sys.stderr)
            return None

        with open(file_path, "r") as f:
            for line in f:
                if "Average success" in line:
                    # Extract the number using regex
                    match = re.search(r"Average success\s+([\d.]+)", line)
                    if match:
                        success_rates.append(float(match.group(1)))

        if not success_rates:
            print(f"Warning: No success rates found in: {file_path}", file=sys.stderr)
            return None

        return success_rates
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}", file=sys.stderr)
        return None


def calculate_average(success_rates):
    if not success_rates:
        return None
    return sum(success_rates) / len(success_rates)


def calculate_overall_average(results):
    """Calculate the overall average success rate from task results"""
    valid_rates = []
    
    for task, rate in results.items():
        if rate is not None:  # Only include tasks with valid results
            valid_rates.append(rate)
    
    if not valid_rates:
        return None
    
    return sum(valid_rates) / len(valid_rates)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Summarize evaluation results from log files."
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Directory containing the log files (e.g., eval_step_30000_gr)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory not found: {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    # Define the tasks in the required order
    tasks = [
        "openvla_pick_coke_can_visual_matching_gpu0",
        "openvla_move_near_visual_matching_gpu1",
        "openvla_put_in_drawer_visual_matching_gpu2",
        "openvla_drawer_visual_matching_gpu3"
    ]

    # Dictionary to store results
    results = {}
    sample_counts = {}
    errors = {}

    # Process each task
    for task in tasks:
        log_file = os.path.join(args.log_dir, f"{task}.log")
        success_rates = extract_success_rates(log_file)

        if success_rates is None:
            results[task] = None
            sample_counts[task] = 0
            errors[task] = "Failed to process log file"
        else:
            avg_success = calculate_average(success_rates)
            results[task] = avg_success
            sample_counts[task] = len(success_rates)
            errors[task] = None

    # Calculate overall average success rate
    overall_avg = calculate_overall_average(results)

    # Write summary to file
    with open(os.path.join(args.log_dir, "evaluation_summary.txt"), "w") as f:
        f.write("Evaluation Results Summary\n")
        f.write("========================\n\n")
        
        # Write individual task results
        for task in tasks:
            f.write(f"{task}:\n")
            if errors[task]:
                f.write(f"  Status: Error - {errors[task]}\n")
            else:
                f.write(f"  Average success: {results[task]:.4f}\n")
                f.write(f"  Number of samples: {sample_counts[task]}\n")
            f.write("\n")
        
        # Write overall average
        f.write("Overall average success rate: ")
        if overall_avg is None:
            f.write("N/A (no valid task data)\n")
        else:
            f.write(f"{overall_avg:.4f}\n")
            print(f"Calculated overall average success rate: {overall_avg:.4f}")


if __name__ == "__main__":
    main()