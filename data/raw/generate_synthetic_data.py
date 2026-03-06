import pandas as pd
import random

def generate_reflection(row):
    """
    Generate a synthetic text reflection based on tabular features.
    """
    risk = row['mental_health_risk']
    stress = row['stress_level']
    anxiety = row['anxiety_score']
    
    # Templates for different risk levels
    high_templates = [
        "I feel extremely overwhelmed lately. My stress level is around {stress} and I'm constantly anxious.",
        "Everything feels like too much. I have high anxiety and it's affecting my daily life.",
        "I'm struggling to cope with the pressure. My anxiety score is {anxiety} and I feel very stressed.",
        "I feel like I'm on the verge of a breakdown. The stress is unbearable and my anxiety is through the roof."
    ]
    
    medium_templates = [
        "I'm feeling a bit pressured recently. My stress level is {stress}, which is manageable but concerning.",
        "I have some anxiety about my situation. It's not constant, but I feel it often.",
        "I'm feeling moderately stressed. Some days are harder than others.",
        "I'm trying to stay calm, but I feel some underlying tension and nervousness."
    ]
    
    low_templates = [
        "I feel quite relaxed and in control. My stress level is only {stress}.",
        "I'm doing well mentally. My anxiety is very low and I feel centered.",
        "I feel calm and prepared for whatever comes next. I don't feel much stress at all.",
        "Everything is going smoothly. I feel mentally balanced and peaceful."
    ]
    
    if risk == 'High':
        template = random.choice(high_templates)
    elif risk == 'Medium':
        template = random.choice(medium_templates)
    else:
        template = random.choice(low_templates)
    
    return template.format(stress=stress, anxiety=anxiety)

def main():
    input_path = "data/raw/mental_health_dataset.csv"
    output_path = "data/raw/mental_health_dataset_with_text.csv"
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Generating synthetic text reflections...")
    df['text'] = df.apply(generate_reflection, axis=1)
    
    # Rename mental_health_risk to anxiety_level for consistency with preprocess.py
    # Also map 'Medium' to 'Moderate'
    df['anxiety_level'] = df['mental_health_risk'].replace('Medium', 'Moderate')
    
    print(f"Saving augmented data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
