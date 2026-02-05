import os
from typing import Union
import json
import regex as re
import ast
import random

def fix_json(input_str):
    fixed_str = re.sub(r'(\w+):', r'"\1":', input_str)
    
    def format_value(match):
        key, value, comma = match.groups()
        value = value.strip()
        if re.match(r'^-?\d+(\.\d+)?$', value):
            value = f'[{value}]'
        elif re.match(r'^(true|false|null)$', value, re.IGNORECASE):
            pass               
        else:
            value = f'"{value}"'
        return f'{key}: {value}{comma}'
    
    fixed_str = re.sub(r'(".*?"):(.*?)(,|})', format_value, fixed_str)
    
    return fixed_str

def read_file_to_string(file_path):
    """
    Reads the contents of a text file and returns it as a string.

    :param file_path: The path to the text file.
    :return: A string containing the contents of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def read_files_to_string(file_paths):
    """
    Reads the contents of multiple text files and returns them as a single string,
    with each file's contents separated by a newline.

    :param file_paths: A list of paths to text files.
    :return: A string containing the concatenated contents of the files.
    """
    all_contents = []                                          

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                all_contents.append(file.read())
        except FileNotFoundError:
            print(f"The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")

    return "\n".join(all_contents)

def get_file_path(filename: Union[str, os.PathLike], search_from: Union[str, os.PathLike] = "."):
    """
    Search for a file across a directory and return its absolute path.

    Args:
        filename (Union[str, os.PathLike]): The name of the file to search for.
        search_from (Union[str, os.PathLike], optional): The directory from which to start the search. Defaults to ".".

    Returns:
        str: Absolute path to the found file.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    for root, dirs, files in os.walk(search_from):
        for name in files:
            if name == filename:
                return os.path.abspath(os.path.join(root, name))
    raise FileNotFoundError(filename, "not found.")



def verify(s, target_sequence):
    count = s.count(target_sequence)

    return count == 2


def is_int_between_0_and_10(s):
    try:
        num = int(s)
        return 0 <= num <= 10
    except ValueError:
        return False
    
def is_str_a_list_of_ints_0_to_10(s):
    try:
        parsed = ast.literal_eval(s)

        if not isinstance(parsed, list):
            return False

        return all(isinstance(item, int) and 0 <= item <= 10 for item in parsed)

    except (ValueError, SyntaxError):
        return False
    
def is_str_valid_score_format_brackets(s):
    try:
        content = s.strip("[]").split(',')

        length = len(content)

        scores = {}
        for item in content:
            key, value = item.split(':')
            key = key.strip()
            value = int(value.strip())

            if not key.startswith("score") or not 0 <= value <= 10:
                return False

            scores[key] = value

        fetch_words = [f"score{i+1}" for i in range(length)]
        return all(key in scores for key in fetch_words)

    except (ValueError, SyntaxError):
        return False


def mllm_output_to_dict(input_string, give_up_parsing=False):
    """
    Args:
        input_string (str): actually the output of the mllm model to be parsed
        output_file_name (str): The name of the output file.
    """
    if input_string == "rate_limit_exceeded":
        return "rate_limit_exceeded"

    delimiter = '||V^=^V||'

    if input_string.count(delimiter) == 2:
        if not verify(input_string, delimiter):
            print("The required delimiters were not found correctly in the string.")
            return False
        start_index = input_string.find(delimiter) + len(delimiter)
        end_index = input_string.rfind(delimiter)
    else:
        start_index = input_string.find('{')
        end_index = input_string.rfind('}') + 1
        if start_index == -1 or end_index == 0:
            start_index = input_string.find('[')
            end_index = input_string.rfind(']') + 1
            if give_up_parsing:                                
                guessed_value = random.randint(0, 10)
                print(f"Failed to find the json content in the string. Guess a value : {guessed_value}.")
                json_content = {'score': [guessed_value], "reasoning": f"guess_if_cannot_parse | {input_string}"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif re.match(r'^\[\d+, ?\d+\]$', input_string[start_index:end_index]):
                scores = json.loads(input_string[start_index:end_index])
                if not isinstance(scores, list):
                    scores = [scores]
                json_content = {'score': scores, "reasoning": "System: output is simply a list of scores"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif is_int_between_0_and_10(input_string):                               
                scores = [int(input_string)]
                json_content = {'score': scores, "reasoning": "System: output is simply a number"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            else:
                print("Failed to find the json content in the string.")
                return False
    
    if start_index != -1 and end_index != -1 and start_index != end_index:
        json_str = input_string[start_index:end_index].strip()
        json_str = json_str.replace("\n", "")
        try:
            new_data = json.loads(json_str)
            if not isinstance(new_data['score'], list):
                new_data['score'] = [new_data['score']]
        except:
            print("Now fixing: ", json_str)
            try:
                new_data = json.loads(fix_json(json_str))
                return new_data
            except:
                print("Error: Cannot fix", json_str)
                return False
        return new_data
    else:
        print("The required delimiters were not found correctly in the string.")
        return False

def write_entry_to_json_file(input_string, uid, prompt_input, vision_input, output_file_name, give_up_parsing=False):
    """
    Args:
        input_string (str): actually the output of the mllm model to be parsed
        uid (str): The unique identifier for the each item in the test data
        prompt_input (str): The prompt input for the entry. text prompt.
        vision_input (str): The vision input for the entry. image links.
        output_file_name (str): The name of the output file.
    """
    if input_string == "rate_limit_exceeded":
        return "rate_limit_exceeded"

    delimiter = '||V^=^V||'

    if input_string.count(delimiter) == 2:
        if not verify(input_string, delimiter):
            print("The required delimiters were not found correctly in the string.")
            return False
        start_index = input_string.find(delimiter) + len(delimiter)
        end_index = input_string.rfind(delimiter)
    else:
        start_index = input_string.find('{')
        end_index = input_string.rfind('}') + 1
        if start_index == -1 or end_index == 0:
            start_index = input_string.find('[')
            end_index = input_string.rfind(']') + 1
            if give_up_parsing:                                
                guessed_value = random.randint(0, 10)
                print(f"Failed to find the json content in the string. Guess a value : {guessed_value}.")
                json_content = {'score': [guessed_value], "reasoning": f"guess_if_cannot_parse | {input_string}"}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif re.match(r'^\[\d+, ?\d+\]$', input_string[start_index:end_index]):
                scores = json.loads(input_string[start_index:end_index])
                json_content = {'score': scores, "reasoning": None}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            elif is_int_between_0_and_10(input_string):                               
                scores = [int(input_string)]
                json_content = {'score': scores, "reasoning": None}
                json_str = json.dumps(json_content)
                input_string = json_str
                start_index = 0
                end_index = len(json_str)
            else:
                print("Failed to find the json content in the string.")
                return False
    
    if start_index != -1 and end_index != -1 and start_index != end_index:
        json_str = input_string[start_index:end_index].strip()
        json_str = json_str.replace("\n", "")
        try:
            new_data = json.loads(json_str)
            
            os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
            
            if os.path.exists(output_file_name):
                with open(output_file_name, 'r') as json_file:
                    data = json.load(json_file)
            else:
                data = {}
            
            if uid in data:
                data[uid].update(new_data)                        
                if prompt_input:                                              
                    data[uid]['prompt_input'] = prompt_input
                if vision_input:                                              
                    data[uid]['vision_input'] = vision_input
            else:
                data[uid] = new_data
                if prompt_input:
                    data[uid]['prompt_input'] = prompt_input
                if vision_input:
                    data[uid]['vision_input'] = vision_input
            
            with open(output_file_name, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            
            print(f"Data was successfully updated in {output_file_name}")
            return True
        except json.JSONDecodeError as e:
            print(f"An error occurred while parsing the JSON content: {e}")
            return False
    else:
        print("The required delimiters were not found correctly in the string.")
        return False


def check_key_in_json(file_path, key):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        
        if key in data:
            return True
        else:
            return False
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error reading {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred with {file_path}: {e}")
    return False