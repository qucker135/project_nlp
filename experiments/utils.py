import pandas as pd
import re

def read_race_dataset(path):
    df_raw = pd.read_parquet(path)
    df_expanded = pd.DataFrame(df_raw['options'].tolist(), columns=['A', 'B', 'C', 'D'])
    df_final = df_raw.drop('options', axis=1).join(df_expanded)
    return df_final

# USE THIS FUNCTION TO READ THE DATASET FROM TXT FILE
def read_mctest_dataset(path):
    with open(path, 'r') as f:
        content = f.read()
    # print(content)
    stories = re.split((r'\*' * 51) + '\n', content)
    stories = [q[(q.find('\n\n')):].strip() for q in stories if q != '']

    rows = []
    for story in stories:
        questions = []
        for _ in range(4): # assuming 4 questions per story
            ind = story.rfind('\n\n')
            question = story[ind:].strip()
            questions.insert(0, question)
            story = story[:ind].strip()
        # print(story[-30:])
        # print(questions)
        assert len(questions) == 4
        
        for question in questions:
            answers = []
            good_answer = None
            for _ in range(4): # assuming 4 answers per question
                ind = question.rfind('\n')
                answer = question[ind:].strip()
                if answer[0] == '*':
                    good_answer = answer[1]
                    answers.insert(0, answer[4:])
                else:
                    answers.insert(0, answer[3:])
                # answers.insert(0, answer)
                question = question[:ind].strip()
            assert len(answers) == 4
            assert good_answer is not None
            # find second ":" in question:
            for _ in range(2):
                question = question[question.find(':')+1:].strip()
            row = {
                'story': story,
                'question': question,
                'A': answers[0],
                'B': answers[1],
                'C': answers[2],
                'D': answers[3],
                'good_answer': good_answer
            }
            rows.append(row)
    return pd.DataFrame(rows)