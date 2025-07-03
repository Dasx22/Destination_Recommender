import pandas as pd
import os

def convert_csv_to_excel(csv_file_path, excel_file_path=None):
    # Check if file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"{csv_file_path} does not exist.")

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # If no output path provided, replace .csv with .xlsx
    if excel_file_path is None:
        excel_file_path = csv_file_path.replace(".csv", ".xlsx")

    # Save to Excel
    df.to_excel(excel_file_path, index=False)
    print(f"✅ Converted '{csv_file_path}' to '{excel_file_path}' successfully!")

if __name__ == "__main__":
    input_csv = "destinations.csv"
    convert_csv_to_excel(input_csv)
    
    input_csv = "user_history.csv"
    convert_csv_to_excel(input_csv)
