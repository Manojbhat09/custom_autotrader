import os
import re
import sys

def extract_local_modules(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    local_modules = set()
    # Regular expression to match lines of the form: from scripts.module_name import ...
    local_module_pattern = re.compile(r"from scripts\.([a-zA-Z_]\w*) import")

    for line in data:
        match = local_module_pattern.search(line)
        if match:
            local_modules.add(match.group(1))

    return local_modules

def write_to_file(modules, output_file):
    with open(output_file, 'w') as out_file:
        out_file.write(f"Local Script Modules:\n")
        out_file.write("="*40 + "\n")
        for module in modules:
            script_file = f'scripts/{module}.py'
            if os.path.exists(script_file):
                with open(script_file, 'r') as script_file_handle:
                    out_file.write(f"Module: {module}\n")
                    out_file.write("-"*40 + "\n")
                    out_file.write(script_file_handle.read() + "\n\n")
            else:
                out_file.write(f"Module: {module} does not exist.\n\n")

def main():
    # Path to the main script file
    main_script_path = 'train_crypto.py'
    
    # Output file name
    output_file = 'local_modules.txt'
    
    # Extracting local modules
    local_modules = extract_local_modules(main_script_path)
    
    # Writing to file
    write_to_file(local_modules, output_file)
    
    print(f'Local script modules written to {output_file}')

if __name__ == "__main__":
    main()
