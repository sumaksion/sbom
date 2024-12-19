import re
import networkx as nx

def process_graph_file(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r') as infile:
            lines = infile.readlines()
        
        with open(output_file_path, 'w') as outfile:
            for line in lines:
                if line.strip().startswith('"') and "[label" in line:
                    modified_content = string_stripper(line)   
                    outfile.write(modified_content + '\n')
                else:
                    outfile.write(line)
        
        print(f"Processed file saved as '{output_file_path}'")
    
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def string_stripper(line: str):
    
    line_parts = line.split("[label = <")
    before_label_content = line_parts[0] + "[label = <" # Everything up to "label = <"
    label_content = line_parts[1].split(">")[0]  
                        
    if (
        label_content.startswith("(IDENTIFIER") or 
        label_content.startswith("(&lt;operator&gt;") or 
        label_content.startswith("(LITERAL") 
        #re.search(r',\w+\.', label_content)  # Matches ",*."
    ):
        modified_content = before_label_content + label_content.split(",")[0] + ")> ]"
    else:

        modified_content = before_label_content + label_content.split()[0].split(")")[0] + ")> ]"
    
    # Replace numbers following 'stack' with zeros
    modified_content = re.sub(
        r'(\$stack)(\d+)',
        lambda m: f'{m.group(1)}{"0" * len(m.group(2))}',
        modified_content
                )
    return modified_content

def get_ast_first_children(dot_file_path):
    extracted_data = []
    with open(dot_file_path, 'r') as file:
        lines = file.readlines()

    is_in_block = False
    is_in_method_return = False
    for line in lines[2:]:

        if line.strip().startswith('"') and "[label" in line:
            match = string_stripper(line.split('" ')[1].strip())
            if "BLOCK" in match:
                if is_in_block:
                    extracted_data.append(match)
                else:
                    extracted_data.append(match)
                    is_in_block = True
            elif "METHOD_RETURN" in match:
                if is_in_method_return:
                    extracted_data.append(match)
                else:
                    extracted_data.append(match)
                    is_in_method_return = True
            elif is_in_block or is_in_method_return:
                break  

            else:
                extracted_data.append(match)    
        else:
            return extracted_data      


    return extracted_data
