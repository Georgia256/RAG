import pandas as pd

def read_questions_answers(file_path):
    questions = []
    answers = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            question, answer = line.strip().split(', ', 1)
            questions.append(question)
            answers.append(answer)
    return questions, answers

def read_context_texts(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    contexts = content.split('~')
    return [context.strip() for context in contexts]

questions_file_path = './Data/harry-potter-questions.txt'
texts_file_path = './Data/harry-potter-texts.txt'

# Reading data from the files
questions, answers = read_questions_answers(questions_file_path)
contexts = read_context_texts(texts_file_path)

# Creating the DataFrame
data = {
    'Question': questions,
    'Answer': answers,
    'Context': contexts[:len(answers)]  # To ensure the lengths match
}

df = pd.DataFrame(data)

try:
    output_csv_path = 'harry-potter-data.csv'
    df.to_csv(output_csv_path, index=False)
    print("harry-potter-data.csv was created successfully")
except:
    print("There was a problem with the creaion of harry-potter-data.csv")
