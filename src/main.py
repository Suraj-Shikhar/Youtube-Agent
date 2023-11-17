from transformers import pipeline , AutoModelForCausalLM, AutoTokenizer
import torch
def load_context(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        context = file.read()
    return context

def main(file_path):
    context = load_context(file_path)
    print(context)
    while True:
        print("\nOptions:")
        print("1. Ask a question w/o time")
        print("2. Exit")
        print("3. ASK QUESTION with time")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            model_name = "deepset/roberta-base-squad2"
            nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
            question = input("Ask your question: ")
            QA_input = {'question': question, 'context': context}
            res = nlp(QA_input)
            print("Answer:", res['answer'])

        elif choice == '2':
            print("Exiting...")
            break

        elif choice == '3':
            question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
            question = input("Ask your question: ")
            result = question_answerer(question=question,context=context)
            print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main("videos/KentaYamashitaTEDxKagoshima.txt")
