import os, sys

def print_code_to_text(files, output_file):
    with open(output_file, 'w') as out_file:
        for file in files:
            if os.path.exists(file):
                with open(file, 'r') as code_file:
                    out_file.write(f"File: {file}\n")
                    out_file.write("="*40 + "\n")
                    out_file.write(code_file.read() + "\n\n")
            else:
                out_file.write(f"File: {file} does not exist.\n\n")

if __name__ == "__main__":
    # List of code files to print out
    code_files = [
        '../streaming/app3.py',
        '../streaming/auth_manager.py',
        '../streaming/data_manager.py',
        '../streaming/login_app/app.py',
        '../streaming/ngrok_script.py',
    ]
    root = os.path.dirname(os.path.abspath(sys.argv[0]))
    for i, file in enumerate(code_files):
        code_files[i] = os.path.join(root, file)
    
    # Name of the output text file
    output_file_name = 'code_output.txt'
    
    print_code_to_text(code_files, output_file_name)
    print(f'Code has been printed to {output_file_name}')
