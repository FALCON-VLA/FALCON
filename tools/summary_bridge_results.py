import re
import argparse

def calculate_average_success_rate(data):
    """
    Calculate the average success rate from the environment summary data.
    
    Args:
        data (str): Multiline string containing environment success summary
        
    Returns:
        float: Average success rate of all tasks, or None if no valid data
    """
    # Regex to match success rate values (including scientific notation)
    rate_regex = re.compile(r':\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    
    success_rates = []
    lines = data.strip().split('\n')
    
    for line in lines:
        # Skip header and empty lines
        if line.startswith('---') or not line.strip():
            continue
            
        # Use regex to find success rate
        match = rate_regex.search(line)
        if match:
            try:
                rate = float(match.group(1))
                success_rates.append(rate)
            except (ValueError, TypeError):
                # Skip invalid conversions
                continue
    
    if not success_rates:
        return None
        
    return sum(success_rates) / len(success_rates)

def process_log_file(input_log_path, output_log_path):
    """
    Parses a log file to extract environment names and their final average success rates,
    then calculates and adds the overall average success rate.
    """
    results = {}
    env_name_regex = re.compile(r"env_name='([^']*)'")
    # Regex to match success rate values (including scientific notation)
    success_rate_regex = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")

    try:
        with open(input_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Successfully read {len(lines)} lines from {input_log_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_log_path}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Iterate over the lines with their index
    for i, line in enumerate(lines):
        if "Average success" in line:
            if i > 0:
                prev_line = lines[i - 1]
                match = env_name_regex.search(prev_line)
                
                if match:
                    env_name = match.group(1)
                    # Extract success rate using regex
                    rate_match = success_rate_regex.search(line.strip())
                    
                    if rate_match:
                        try:
                            success_rate = float(rate_match.group(1))
                            results[env_name] = success_rate
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid success rate format in line: '{line.strip()}'. Error: {e}")
                    else:
                        print(f"Warning: No valid success rate found in line: '{line.strip()}'")
                else:
                    print(f"Warning: Found 'Average success' but no 'env_name' in line: '{prev_line.strip()}'")

    try:
        with open(output_log_path, "w", encoding="utf-8") as f:
            f.write("--- Environment Success Summary ---\n")
            
            if not results:
                f.write("No valid 'Average success' entries found.\n")
                print("Processing complete. No valid results found.")
                return
            
            # Write each environment and its success rate
            for env_name, success_rate in results.items():
                f.write(f"{env_name}: {success_rate}\n")
            
            # Calculate and add overall average success rate
            summary_data = "\n".join([f"{k}: {v}" for k, v in results.items()])
            overall_avg = calculate_average_success_rate(summary_data)
            
            if overall_avg is None:
                f.write("\nCould not calculate overall average (no valid rates)\n")
                print("Warning: No valid rates for overall average calculation")
            else:
                f.write(f"\nOverall average success rate: {overall_avg:.4f}\n")
                print(f"Calculated overall average success rate: {overall_avg:.4f}")
        
        print(f"Processing complete. Summary written to '{output_log_path}'")
    except Exception as e:
        print(f"Critical error writing output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a log file to extract environment success rates and calculate overall average."
    )
    parser.add_argument("input_file", type=str, help="Path to the input log file.")
    
    args = parser.parse_args()
    
    # Generate output path with '_summary' suffix
    if '.' in args.input_file:
        base, ext = args.input_file.rsplit('.', 1)
        output_file_path = f"{base}_summary.{ext}"
    else:
        output_file_path = f"{args.input_file}_summary"
    
    process_log_file(args.input_file, output_file_path)