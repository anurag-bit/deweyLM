#!/usr/bin/env python3
"""
Fine-tuning Dataset Preparation for Gemma-3-270M-IT
Prepares the Dewey Classification text for fine-tuning using various strategies.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import re

class DeweyFineTuningDatasetCreator:
    def __init__(self, processed_data_dir: str = "processed_data", 
                 output_dir: str = "fine_tuning_data"):
        self.processed_data_dir = Path(processed_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load processed data
        self.chunks = self.load_chunks()
        self.full_text = self.load_full_text()
        
    def load_chunks(self) -> List[Dict]:
        """Load processed text chunks."""
        chunks_file = self.processed_data_dir / "text_chunks.json"
        with open(chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_full_text(self) -> str:
        """Load full processed text."""
        text_file = self.processed_data_dir / "dewey_classification_full_text.txt"
        with open(text_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_classification_patterns(self) -> List[Dict]:
        """Extract classification number patterns and descriptions."""
        patterns = []
        
        # Pattern for Dewey decimal numbers (XXX.XXX)
        dewey_pattern = r'(\d{3}(?:\.\d+)?)\s+([^\n]+)'
        
        matches = re.findall(dewey_pattern, self.full_text)
        
        for number, description in matches:
            if len(description.strip()) > 10:  # Filter out short matches
                patterns.append({
                    'classification_number': number,
                    'description': description.strip(),
                    'type': 'classification'
                })
        
        return patterns
    
    def create_qa_pairs(self) -> List[Dict]:
        """Create question-answer pairs from the text."""
        qa_pairs = []
        
        # Extract classification patterns first
        classifications = self.extract_classification_patterns()
        
        # Generate different types of questions
        question_templates = [
            "What is the Dewey Decimal Classification number for {}?",
            "What does the classification number {} represent?",
            "What subject is classified under {}?",
            "How is {} classified in the Dewey Decimal system?",
            "What classification number is used for {}?"
        ]
        
        for cls in classifications[:500]:  # Limit for quality
            number = cls['classification_number']
            description = cls['description']
            
            # Create multiple QA pairs for each classification
            qa_pairs.extend([
                {
                    'question': f"What does the classification number {number} represent?",
                    'answer': description,
                    'type': 'number_to_description'
                },
                {
                    'question': f"What is the Dewey Decimal Classification for {description.lower()}?",
                    'answer': f"The classification number is {number}",
                    'type': 'description_to_number'
                }
            ])
        
        return qa_pairs
    
    def create_instruction_dataset(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Create instruction-following dataset in Alpaca format."""
        instruction_data = []
        
        system_prompts = [
            "You are an expert librarian specializing in the Dewey Decimal Classification system.",
            "You are a knowledgeable assistant helping users understand library classification systems.",
            "You are an expert in organizing and categorizing knowledge using the Dewey Decimal system."
        ]
        
        for qa in qa_pairs:
            instruction_data.append({
                'instruction': qa['question'],
                'input': '',
                'output': qa['answer'],
                'system': random.choice(system_prompts)
            })
        
        return instruction_data
    
    def create_conversational_dataset(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Create conversational dataset in chat format."""
        conversations = []
        
        for qa in qa_pairs:
            conversation = {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert librarian specializing in the Dewey Decimal Classification system. Provide accurate and helpful information about library classification.'
                    },
                    {
                        'role': 'user',
                        'content': qa['question']
                    },
                    {
                        'role': 'assistant',
                        'content': qa['answer']
                    }
                ]
            }
            conversations.append(conversation)
        
        return conversations
    
    def create_completion_dataset(self) -> List[Dict]:
        """Create text completion dataset from chunks."""
        completions = []
        
        for chunk in self.chunks[:1000]:  # Limit for quality
            text = chunk['text']
            if len(text.split()) > 50:  # Ensure reasonable length
                # Split text for completion task
                words = text.split()
                split_point = len(words) // 2
                
                prompt = ' '.join(words[:split_point])
                completion = ' '.join(words[split_point:])
                
                completions.append({
                    'prompt': f"Continue this text about the Dewey Decimal Classification: {prompt}",
                    'completion': completion,
                    'source_page': chunk.get('source_page', 'unknown')
                })
        
        return completions
    
    def create_classification_training_data(self) -> List[Dict]:
        """Create specific classification training examples."""
        training_data = []
        
        # Extract hierarchical structure
        main_classes = {}
        divisions = {}
        sections = {}
        
        # Parse the text for hierarchical information
        lines = self.full_text.split('\n')
        current_class = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for main class headers (000-099, 100-199, etc.)
            main_class_match = re.match(r'(\d)00[s]?\s*[-–]\s*(\d)99[s]?\s+(.+)', line)
            if main_class_match:
                class_range = f"{main_class_match.group(1)}00s"
                description = main_class_match.group(3)
                main_classes[class_range] = description
                current_class = class_range
                continue
            
            # Look for specific classifications
            class_match = re.match(r'(\d{3}(?:\.\d+)?)\s+(.+)', line)
            if class_match:
                number = class_match.group(1)
                description = class_match.group(2)
                
                training_data.extend([
                    {
                        'input': f"Classify: {description}",
                        'output': f"Dewey Classification: {number}",
                        'type': 'classification_task'
                    },
                    {
                        'input': f"What is {number} in the Dewey system?",
                        'output': description,
                        'type': 'lookup_task'
                    }
                ])
        
        return training_data
    
    def generate_all_datasets(self):
        """Generate all types of training datasets."""
        print("Extracting classification patterns...")
        classifications = self.extract_classification_patterns()
        print(f"Found {len(classifications)} classification patterns")
        
        print("Creating QA pairs...")
        qa_pairs = self.create_qa_pairs()
        print(f"Created {len(qa_pairs)} QA pairs")
        
        print("Creating instruction dataset...")
        instruction_data = self.create_instruction_dataset(qa_pairs)
        
        print("Creating conversational dataset...")
        conversational_data = self.create_conversational_dataset(qa_pairs)
        
        print("Creating completion dataset...")
        completion_data = self.create_completion_dataset()
        
        print("Creating classification training data...")
        classification_data = self.create_classification_training_data()
        
        # Save all datasets
        datasets = {
            'instruction_dataset': instruction_data,
            'conversational_dataset': conversational_data,
            'completion_dataset': completion_data,
            'classification_dataset': classification_data,
            'qa_pairs': qa_pairs,
            'raw_classifications': classifications
        }
        
        for name, data in datasets.items():
            if data:  # Only save non-empty datasets
                output_file = self.output_dir / f"{name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(data)} items to {output_file}")
        
        # Create combined training dataset
        combined_training = []
        
        # Add instruction data
        for item in instruction_data:
            combined_training.append({
                'text': f"<|system|>\n{item['system']}\n<|user|>\n{item['instruction']}\n<|assistant|>\n{item['output']}<|end|>",
                'type': 'instruction'
            })
        
        # Add completion data
        for item in completion_data[:200]:  # Limit completion data
            combined_training.append({
                'text': f"<|user|>\n{item['prompt']}\n<|assistant|>\n{item['completion']}<|end|>",
                'type': 'completion'
            })
        
        # Save combined dataset
        combined_file = self.output_dir / "combined_training_dataset.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_training, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(combined_training)} items to combined training dataset")
        
        # Generate statistics
        stats = {
            'total_classifications': len(classifications),
            'total_qa_pairs': len(qa_pairs),
            'instruction_examples': len(instruction_data),
            'conversational_examples': len(conversational_data),
            'completion_examples': len(completion_data),
            'classification_examples': len(classification_data),
            'combined_training_examples': len(combined_training)
        }
        
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print("\nDataset Statistics:")
        print("-" * 30)
        for key, value in stats.items():
            print(f"{key}: {value:,}")
        
        return datasets, stats

def main():
    """Generate fine-tuning datasets."""
    processed_data_dir = Path("processed_data")
    
    if not processed_data_dir.exists():
        print("Processed data not found. Please run process_pdf_server.py first.")
        return
    
    print("Creating fine-tuning datasets...")
    creator = DeweyFineTuningDatasetCreator()
    
    datasets, stats = creator.generate_all_datasets()
    
    print("\nFine-tuning datasets created successfully!")
    print("Available datasets:")
    print("- instruction_dataset.json (Alpaca-style)")
    print("- conversational_dataset.json (Chat format)")
    print("- completion_dataset.json (Text completion)")
    print("- classification_dataset.json (Classification tasks)")
    print("- combined_training_dataset.json (All formats combined)")
    
    print("\nNext steps:")
    print("1. Use these datasets with LoRA/QLoRA for parameter-efficient fine-tuning")
    print("2. Consider using PEFT (Parameter-Efficient Fine-Tuning) methods")
    print("3. Monitor training with appropriate validation metrics")

if __name__ == "__main__":
    main()